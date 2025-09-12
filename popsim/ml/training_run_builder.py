from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import jax.numpy as jnp
import optax
import xarray as xr
from jaxtyping import PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import DataLoader
from popsim.ml.eval import EvaluationSuite


class TrainRunBuilder(ABC):
    """Abstract base class for defining training run builders."""

    @staticmethod
    @abstractmethod
    def get_dataloaders(dataloader_config: dict) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Get the dataset and dataloaders for train/val/test."""

    @staticmethod
    @abstractmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

    @staticmethod
    @abstractmethod
    def get_loss_fn(config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """
        Return a function loss_fn(prediction, target) -> scalar loss.
        """

    @staticmethod
    @abstractmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        """
        Return an Optax optimizer/transform given the config.
        """

    @staticmethod
    def get_val_eval_suite(config: dict) -> EvaluationSuite:
        """Optionally return your train time evaluation suite."""
        return None

    @staticmethod
    def get_test_eval_suite(config: dict) -> EvaluationSuite:
        """Optionally return your test time evaluation suite."""
        return None

    @staticmethod
    def get_trainable_getter(config: dict) -> Optional[Callable[[TrainableModel], PyTree]]:
        """Optionally return a function that takes in the trainable parameters of your model and returns the trainable parameters."""
        return None
