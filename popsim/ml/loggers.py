import json
from abc import ABC, abstractmethod
from collections.abc import Callable

import loguru
import numpy as np
from jaxtyping import Array

from popsim.utils import flatten_dict

"""
Logging utilities for tracking training progress.
TODO(allenw): not much time was spent on this, it could use considerable improvement.
"""

# Run summary key set when a sweep stops the run before it trains to completion (e.g. hyperband pruning)
STOP_REQUESTED_SUMMARY_KEY = "stop_requested"


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
    # Log train metrics only on validation epochs instead of every epoch.
    log_at_val_cadence: bool = False

    @abstractmethod
    def log(self, dictionary):
        raise NotImplementedError

    def stop_requested(self) -> bool:
        """Whether an external controller asked training to stop at the next epoch boundary."""
        return False


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
    # Per-epoch logs from fast-training models overload the W&B backend.
    log_at_val_cadence = True

    def __init__(self, run, run_should_stop: Callable[[str], bool] | None = None):
        """
        Args:
            run: The W&B run to log to.
            run_should_stop (Callable[[str], bool] | None, optional): Called with the run id at each epoch boundary,
                True when the sweep agent was told to stop this run. Defaults to None (never stop).
        """
        self.run = run
        self.run_should_stop = run_should_stop
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("val/*", step_metric="val/epoch")

    def log(self, dictionary):
        self.run.log(flatten_dict(dictionary))

    def stop_requested(self) -> bool:
        if self.run_should_stop is None:
            return False
        should_stop = self.run_should_stop(self.run.id)
        # A cooperatively stopped run still ends "finished" on the WandB server,
        # so the summary is the only record that it did not train to completion.
        if should_stop:
            self.run.summary[STOP_REQUESTED_SUMMARY_KEY] = True
        return should_stop


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
