"""Live W&B checks that a sweep stop ends a trial at an epoch boundary and leaves the agent healthy.

These talk to the real W&B server, so they are opt-in with POPSIM_LIVE_WANDB_TESTS=1 and need an API key.
Each test drives launch_agent in a subprocess, because the agent blocks its main thread
and the Ctrl-C path needs a real SIGINT.
The subprocess output is read live, so a stop can be triggered mid-trial at a known epoch.
The stop itself goes through the same server paths the UI and hyperband use.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import wandb
import xarray as xr
import yaml
from wandb.sdk.internal.internal_api import Api as InternalApi

from popsim.ml.dataloading import make_time_indep_dataloader
from popsim.ml.eval import make_val_loss_eval_fn
from popsim.ml.train_run_builder import TrainRunBuilder

LIVE_TESTS_ENABLED = os.environ.get("POPSIM_LIVE_WANDB_TESTS") == "1" and wandb.api.api_key is not None
pytestmark = pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Needs POPSIM_LIVE_WANDB_TESTS=1 and a W&B API key")

PROJECT = "popsim-sweep-agent-test"
# One epoch is about 1 s on a CPU, so a full trial takes about 25 s and a stop lands within a second.
TRAIN_CONFIG = {
    "project": PROJECT,
    "train_run_builder": "popsim.tests.test_ml.test_sweep_agent.ToyTrainRunBuilder",
    "max_epochs": 20,
    "epochs_per_val": 1,
    "checkpoint_dir": None,
    "dataloader_config": {"seed": 0, "n_train": 20000, "n_eval": 2000, "batch_size": 8},
    "model_init_config": {},
    "loss_config": {},
    "optimizer_config": {"learning_rate": 0.01},
}
# A single-valued parameter would exhaust the sweep after one trial, so sample a range.
SWEEP_CONFIG = {
    "method": "random",
    "metric": {"name": "val/loss.mean", "goal": "minimize"},
    "parameters": {
        "optimizer_config": {"parameters": {"learning_rate": {"distribution": "log_uniform_values", "min": 1.0e-3, "max": 1.0e-1}}}
    },
}
STOP_AT_EPOCH = 2
TRIAL_PROGRESS_TIMEOUT_S = 120.0
AGENT_EXIT_TIMEOUT_S = 180.0
RUN_STATE_SETTLE_TIMEOUT_S = 60.0
STOP_MESSAGE = "Stop requested at epoch"
LAUNCH_AGENT_SCRIPT = """
import sys
from popsim.ml.launch import launch_agent
launch_agent(sys.argv[1], sys.argv[2], {"count": int(sys.argv[3])})
"""
_TRIAL_HEADER = re.compile(r"Agent Starting Run: (\w+)")
_EPOCH_BAR = re.compile(r"\| (\d+)/\d+ \[")


class _LinearModel(eqx.Module):
    slope: jax.Array
    offset: jax.Array

    def __call__(self, inputs):
        y_pred = self.slope * inputs["x"] + self.offset
        return {"y": y_pred}


def _squared_error(prediction, target):
    residual = prediction["y"] - target["y"]
    return jnp.square(residual)


def _noisy_line_split(n_samples: int, rng: np.random.Generator) -> xr.Dataset:
    x = rng.uniform(-1.0, 1.0, size=(1, n_samples))
    noise = rng.normal(0.0, 0.1, size=(1, n_samples))
    y = 3.0 * x - 0.5 + noise
    time_coord = np.arange(n_samples, dtype=float)
    return xr.Dataset(
        {"x": (("episode", "time"), x), "y": (("episode", "time"), y)},
        coords={"episode": [0], "time": time_coord},
    )


class ToyTrainRunBuilder(TrainRunBuilder):
    """Linear regression whose test eval logs test/loss.mean, so a stopped trial is visibly missing it."""

    @staticmethod
    def get_dataloaders(dataloader_config):
        rng = np.random.default_rng(dataloader_config["seed"])
        ds_train = _noisy_line_split(dataloader_config["n_train"], rng)
        ds_val = _noisy_line_split(dataloader_config["n_eval"], rng)
        ds_test = _noisy_line_split(dataloader_config["n_eval"], rng)
        dl_kwargs = dict(time_coord="time", episode_coord="episode", input_vars=["x"], target_vars=["y"])
        train_dl = make_time_indep_dataloader(
            ds_train, batch_size=dataloader_config["batch_size"], shuffle=True, drop_last=True, **dl_kwargs
        )
        val_dl = make_time_indep_dataloader(ds_val, shuffle=False, **dl_kwargs)
        test_dl = make_time_indep_dataloader(ds_test, shuffle=False, **dl_kwargs)
        return ds_train, train_dl, val_dl, test_dl

    @staticmethod
    def model_init(train_dl, model_init_config):
        return _LinearModel(slope=jnp.array(0.0), offset=jnp.array(0.0))

    @staticmethod
    def get_loss_fn(config):
        return _squared_error

    @staticmethod
    def get_optimizer(config):
        return optax.sgd(config["learning_rate"])

    @staticmethod
    def get_test_eval_suite(config):
        return {"loss": make_val_loss_eval_fn(_squared_error)}


@dataclass
class _Sweep:
    entity: str
    sweep_id: str
    cancelled: bool = field(default=False, init=False)

    def run_path(self, run_id: str) -> str:
        return f"{self.entity}/{PROJECT}/{run_id}"

    def cancel(self):
        if self.cancelled:
            return
        InternalApi().cancel_sweep(self.sweep_id, entity=self.entity, project=PROJECT)
        self.cancelled = True

    def stop_run(self, run_id: str):
        """The server path behind the UI stop button and hyperband, delivered to the agent as a stop command."""
        run = wandb.Api().run(self.run_path(run_id))
        assert InternalApi().stop_run(run.storage_id)

    def run_summary(self, run_id: str) -> tuple[str, dict]:
        """The run's state and summary once the server has moved it out of running."""
        deadline = time.monotonic() + RUN_STATE_SETTLE_TIMEOUT_S
        while True:
            # A fresh Api each poll, the public Api caches run objects.
            run = wandb.Api().run(self.run_path(run_id))
            if run.state != "running" or time.monotonic() > deadline:
                return run.state, dict(run.summary)
            time.sleep(2.0)


class _AgentProcess:
    """launch_agent in a subprocess, with its output collected live so tests can act mid-trial."""

    def __init__(self, command: list[str], cwd: str, env: dict):
        self.process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._chunks = []
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self):
        fd = self.process.stdout.fileno()
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                return
            self._chunks.append(chunk)

    @property
    def output(self) -> str:
        return b"".join(self._chunks).decode(errors="replace")

    def wait_for_trial_epoch(self, trial_index: int, epoch: int) -> str:
        """Block until trial number trial_index (0-based) has trained epoch epochs and return its run id.

        Progress is read from the tqdm epoch bar of that trial's section of the output.
        """
        deadline = time.monotonic() + TRIAL_PROGRESS_TIMEOUT_S
        while time.monotonic() < deadline:
            output = self.output
            headers = list(_TRIAL_HEADER.finditer(output))
            if len(headers) > trial_index:
                section_start = headers[trial_index].end()
                section_end = headers[trial_index + 1].start() if len(headers) > trial_index + 1 else len(output)
                epochs_done = [int(match.group(1)) for match in _EPOCH_BAR.finditer(output[section_start:section_end])]
                if epochs_done and max(epochs_done) >= epoch:
                    return headers[trial_index].group(1)
            if self.process.poll() is not None:
                pytest.fail(f"Agent exited before trial {trial_index} reached epoch {epoch}:\n{output}")
            time.sleep(0.2)
        pytest.fail(f"Trial {trial_index} did not reach epoch {epoch} within {TRIAL_PROGRESS_TIMEOUT_S} s:\n{self.output}")

    def send_sigint(self):
        self.process.send_signal(signal.SIGINT)

    def finish(self) -> str:
        self.process.wait(timeout=AGENT_EXIT_TIMEOUT_S)
        self._reader.join(timeout=10.0)
        return self.output

    def kill_if_running(self):
        if self.process.poll() is None:
            self.process.kill()


@pytest.fixture
def sweep():
    entity = wandb.Api().default_entity
    sweep_id = wandb.sweep(SWEEP_CONFIG, entity=entity, project=PROJECT)
    handle = _Sweep(entity, sweep_id)
    yield handle
    handle.cancel()


@pytest.fixture
def agent_process(sweep, tmp_path):
    """Hand back a factory that starts launch_agent on the sweep, so tests choose the trial count."""
    config_path = tmp_path / "train_config.yaml"
    with open(config_path, "w") as config_file:
        yaml.safe_dump(TRAIN_CONFIG, config_file)
    env = dict(os.environ) | {"WANDB_DIR": str(tmp_path), "PYTHONUNBUFFERED": "1"}
    started = []

    def _start(count: int) -> _AgentProcess:
        command = [sys.executable, "-c", LAUNCH_AGENT_SCRIPT, str(config_path), sweep.sweep_id, str(count)]
        agent = _AgentProcess(command, cwd=str(tmp_path), env=env)
        started.append(agent)
        return agent

    yield _start
    for agent in started:
        agent.kill_if_running()


def _assert_clean_agent_exit(agent: _AgentProcess, output: str):
    assert agent.process.returncode == 0, output
    assert "Run threw exception" not in output, output
    assert "Exception in thread" not in output, output


def _assert_stopped_without_test_eval(sweep: _Sweep, run_id: str):
    state, summary = sweep.run_summary(run_id)
    assert state == "finished", state
    assert "test/loss.mean" not in summary
    assert STOP_AT_EPOCH <= summary["train/epoch"] < TRAIN_CONFIG["max_epochs"]


def test_sweep_cancel_stops_at_epoch_boundary(sweep, agent_process):
    agent = agent_process(count=1)
    run_id = agent.wait_for_trial_epoch(0, STOP_AT_EPOCH)
    sweep.cancel()
    output = agent.finish()

    _assert_clean_agent_exit(agent, output)
    assert STOP_MESSAGE in output, output
    _assert_stopped_without_test_eval(sweep, run_id)


def test_run_stop_leaves_next_trial_healthy(sweep, agent_process):
    """The hyperband path: one trial is stopped, the next trial in the same process trains to the test eval."""
    agent = agent_process(count=2)
    stopped_run_id = agent.wait_for_trial_epoch(0, STOP_AT_EPOCH)
    sweep.stop_run(stopped_run_id)
    next_run_id = agent.wait_for_trial_epoch(1, 1)
    output = agent.finish()

    _assert_clean_agent_exit(agent, output)
    assert output.count(STOP_MESSAGE) == 1, output
    _assert_stopped_without_test_eval(sweep, stopped_run_id)
    next_state, next_summary = sweep.run_summary(next_run_id)
    assert next_state == "finished", next_state
    assert "test/loss.mean" in next_summary
    assert next_summary["train/epoch"] == TRAIN_CONFIG["max_epochs"]


def test_sigint_stops_at_epoch_boundary(sweep, agent_process):
    agent = agent_process(count=1)
    run_id = agent.wait_for_trial_epoch(0, STOP_AT_EPOCH)
    agent.send_sigint()
    output = agent.finish()

    _assert_clean_agent_exit(agent, output)
    assert "Stopping sweep after the current epoch" in output, output
    assert STOP_MESSAGE in output, output
    _assert_stopped_without_test_eval(sweep, run_id)
