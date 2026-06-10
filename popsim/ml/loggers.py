import json
import time
from abc import ABC, abstractmethod

import loguru
import numpy as np
from jaxtyping import Array

from popsim.utils import flatten_dict

"""
Logging utilities for tracking training progress.
TODO(allenw): not much time was spent on this, it could use considerable improvement.
"""


def convert_val_to_serializable(val):
    if isinstance(val, Array | np.ndarray):
        return val.tolist()
    return val


def _simplified_repr(dictionary, max_string_size):
    """Return a simplified representation of a dictionary, truncating strings and replacing most objects with their type."""
    truncated_dict = {}
    for key, val in dictionary.items():
        # If the value is a dictionary, recurse
        if isinstance(val, dict):
            truncated_dict[key] = _simplified_repr(val, max_string_size)
        else:
            str_val = str(val)
            if len(str_val) <= max_string_size:
                truncated_dict[key] = convert_val_to_serializable(val)
            else:
                val_type = str(type(val)).split("'")[1]
                truncated_dict[key] = f"{val_type}: {str_val[:max_string_size]}..."
    return truncated_dict


class LoggerBase(ABC):
    @abstractmethod
    def log(self, dictionary):
        raise NotImplementedError


class NullLogger(LoggerBase):
    def log(self, dictionary):
        return


class ConsoleLogger(LoggerBase):
    def __init__(self, max_string_size=50):
        self.max_string_size = max_string_size

    def log(self, dictionary):
        truncated_dict = _simplified_repr(dictionary, self.max_string_size)
        loguru.logger.info(json.dumps(truncated_dict, indent=2))


class WandbLogger(LoggerBase):
    # How often to poll W&B for an early-termination request, in seconds.
    stop_poll_interval_s: float = 30.0

    def __init__(self, run):
        self.run = run
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("val/*", step_metric="val/epoch")
        self._last_stop_poll = time.monotonic()

    def log(self, dictionary):
        self.run.log(flatten_dict(dictionary))
        self._raise_if_stop_requested()

    def _raise_if_stop_requested(self):
        """Exit cooperatively when the W&B sweep controller requests early termination.

        When a sweep controller (e.g. hyperband) terminates a run early, the W&B agent's only
        enforcement mechanism is injecting a bare Exception into the training thread at an
        arbitrary bytecode boundary (wandb.agents.pyagent._terminate_thread). That injection can
        land inside JAX/XLA or HDF5/netCDF calls, leaving locks or caches in a corrupted state
        that destabilizes subsequent runs sharing the agent process. Polling the stop flag here
        lets the run exit at a safe point instead.
        """
        now = time.monotonic()
        if now - self._last_stop_poll < self.stop_poll_interval_s:
            return
        self._last_stop_poll = now
        if self._stop_requested():
            raise KeyboardInterrupt("W&B requested early termination of this run.")

    def _stop_requested(self) -> bool:
        # This uses internal W&B APIs (the public Run object does not expose the stop flag),
        # so fail closed to "keep running" and let the agent's own termination handle it
        # if these APIs change.
        try:
            handle = self.run._interface.deliver_stop_status()
            result = handle.wait_or(timeout=5)
            return bool(result.response.stop_status_response.run_should_stop)
        except Exception:
            return False


def get_logger(logger_type: str, **kwargs) -> LoggerBase:
    """
    Factory function to create a logger based on the specified type.

    Args:
        logger_type (str): The type of logger to create ('null', 'console', or 'wandb').
        **kwargs: Additional arguments to pass to the logger constructor.

    Returns:
        LoggerBase: An instance of the specified logger type.

    Raises:
        ValueError: If an invalid logger type is specified.
    """
    if logger_type == "null":
        return NullLogger()
    elif logger_type == "console":
        return ConsoleLogger(**kwargs)
    elif logger_type == "wandb":
        return WandbLogger(**kwargs)
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")
