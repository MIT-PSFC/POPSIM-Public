from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr

from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.modules.power_balance.module import PowerBalance, PowerBalanceEnv


class PowerBalanceTrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(dataloader_config: dict) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError("For now, PowerBalance is a submodule, should use the main module's dataloaders")

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        # Time-dependent module, need to wrap in an env
        module = PowerBalance.init(model_init_config)
        env = PowerBalanceEnv(module=module, stepper=model_init_config["stepper"])
        return env

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        """
        Return a function loss_fn(prediction, target) -> scalar loss.
        """

        def loss_fn(pred, targ):
            absolute_error = jnp.abs(pred.Wtot_MJ_pred - targ["Wtot_MJ"].data)
            return optax.huber_loss(absolute_error)

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_optimizer(optimizer_config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=optimizer_config["lr0"],
            transition_steps=optimizer_config["transition_steps"],
            decay_rate=optimizer_config["decay_rate"],
            end_value=optimizer_config["lrf"],
        )
        opt = optax.adamw(learning_rate=schedule, weight_decay=optimizer_config["weight_decay"])
        return opt
