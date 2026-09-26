import signal
import threading

import wandb
from wandb.agents import pyagent


class CooperativeStopAgent(pyagent.Agent):
    """Sweep agent that records a server stop command instead of killing the run thread.

    pyagent.Agent._stop_run injects a bare Exception into the run thread at an arbitrary bytecode boundary.
    That can land inside JAX/XLA or HDF5 calls and corrupt state for later runs in the same process.
    Here a stop only marks the run, and the Trainer polls run_should_stop at each epoch boundary.

    Ctrl-C is handled the same way.
    pyagent relies on KeyboardInterrupt inside Thread.join for that, which on Python 3.12
    marks the still-running thread as stopped, so the interpreter shuts down underneath it.
    """

    def _init(self):
        super()._init()
        # Kept apart from _run_status, which the job loop overwrites with RUNNING right after starting the thread.
        # A stop landing in that window would otherwise be lost.
        self._stop_requested_run_ids = set()

    def _stop_run(self, run_id):
        self._stop_requested_run_ids.add(run_id)
        self._run_status[run_id] = pyagent.RunStatus.STOPPED

    def run_should_stop(self, run_id: str) -> bool:
        """Whether the sweep server asked this agent to stop the run."""
        return run_id in self._stop_requested_run_ids

    def run(self):
        in_main_thread = threading.current_thread() is threading.main_thread()
        if not in_main_thread:
            super().run()
            return
        previous_handler = signal.signal(signal.SIGINT, self._on_sigint)
        try:
            super().run()
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    def _on_sigint(self, signum, frame):
        wandb.termlog("Ctrl + C detected. Stopping sweep after the current epoch, press again to force.")
        signal.signal(signal.SIGINT, signal.default_int_handler)
        self._exit()
