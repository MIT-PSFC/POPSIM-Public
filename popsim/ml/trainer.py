import time
import typing

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax_dataloader import DataLoader
from jaxtyping import Array, PyTree
from tqdm import tqdm

from popsim.ml._types import TrainableModel
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.eval import EvaluationSuite, batch_loss_and_grad, eval_module_on_data, make_val_loss_eval_fn
from popsim.ml.loggers import ConsoleLogger, LoggerBase
from popsim.ml.loss import InstantaneousLoss, IntegralLoss, LossFunction
from popsim.ml.partition import PartitionFn, make_partition_by_members
from popsim.tree_util import any_nans


@chex.dataclass
class TrainState:
    step: int
    epoch: int
    model: TrainableModel
    opt_state: optax.OptState

    @classmethod
    def create_new(
        cls,
        model: TrainableModel,
        partition_fn: PartitionFn,
        optimizer: optax.GradientTransformation,
    ) -> "TrainState":
        trainable, _ = partition_fn(model)
        opt_state = optimizer.init(trainable)
        return cls(step=0, epoch=0, model=model, opt_state=opt_state)


@eqx.filter_jit
def train_step(
    model: TrainableModel,
    partition_fn: PartitionFn,
    loss_fn: IntegralLoss,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    inputs: PyTree[Array],
    targets: PyTree[Array],
) -> tuple[TrainableModel, optax.OptState, float]:
    # Partition the model into trainable and static parts.
    trainable, static = partition_fn(model)

    # Compute the loss value and the gradient of loss w.r.t. the trainable parts of the model.
    loss_value, grads = batch_loss_and_grad(trainable, static, loss_fn, inputs, targets)

    loss_value = eqx.error_if(loss_value, jnp.isnan(loss_value), "Loss is NaN!")
    grads = eqx.error_if(grads, any_nans(grads), "Gradients contain NaNs!")

    # Update the optimizer and the model.
    model_updates, opt_state = optimizer.update(grads, opt_state, trainable)

    model_updates = eqx.error_if(model_updates, any_nans(model_updates), "Model updates contain NaNs!")

    # Apply the updates to the trainable part of the model.
    trainable = eqx.apply_updates(trainable, model_updates)

    # Combine the trainable and static parts of the model to get back the whole model.
    new_model = eqx.combine(trainable, static)
    return new_model, opt_state, loss_value


def train_epoch(
    train_state: TrainState,
    partition_fn: PartitionFn,
    loss_fn: InstantaneousLoss,
    optimizer: optax.GradientTransformation,
    train_dl: DataLoader,
    logger: LoggerBase,
) -> TrainState:
    for batch in train_dl:
        inputs, targets = batch.ds.popsim_ml.prep_inputs_and_targets()

        tstart_step = time.time()

        model, opt_state, loss_value = train_step(
            train_state.model,
            partition_fn,
            loss_fn,
            optimizer,
            train_state.opt_state,
            inputs,
            targets,
        )

        tend_step = time.time()

        logger.log(
            {
                "train/loss": loss_value,
                "train/step": train_state.step,
                "train/step_time": tend_step - tstart_step,
            }
        )
        train_state = TrainState(
            step=train_state.step + 1,
            epoch=train_state.epoch,
            model=model,
            opt_state=opt_state,
        )
    train_state.epoch += 1
    return train_state


class Trainer:
    train_state: TrainState
    optimizer: optax.GradientTransformation
    partition_fn: PartitionFn
    loss_fn: LossFunction
    logger: LoggerBase
    checkpoint_manager: ocp.CheckpointManager

    def __init__(
        self,
        model: TrainableModel,
        trainable_getter: typing.Callable[[TrainableModel], PyTree[Array]],
        loss_fn: LossFunction,
        optimizer: optax.GradientTransformation,
        checkpoint_manager: typing.Optional[ocp.CheckpointManager] = None,
    ):
        if isinstance(model, ModuleEvalEnv):
            assert isinstance(loss_fn, IntegralLoss), "When using a ModuleEvalEnv, the loss function must be an IntegralLoss."

        self.partition_fn = make_partition_by_members(trainable_getter)
        self.train_state = TrainState.create_new(model, self.partition_fn, optimizer)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_manager = checkpoint_manager

    def train(
        self,
        train_dl: DataLoader,
        val_dl: typing.Optional[DataLoader],
        eval_suite: typing.Optional[EvaluationSuite] = None,
        max_epochs: int = 1000,
        epochs_per_val: int = 1,
        patience: typing.Optional[int] = None,
        logger: typing.Optional[LoggerBase] = None,
    ):
        logger = logger or ConsoleLogger()
        eval_suite = eval_suite or {}
        if "loss" not in eval_suite:
            eval_suite["loss"] = make_val_loss_eval_fn(self.loss_fn)

        val_loss_history = np.array([])

        # epoch_range accounts for restarting training from a checkpoint.
        epoch_range = range(self.train_state.epoch, self.train_state.epoch + max_epochs + 1)

        for epoch in tqdm(epoch_range, desc="Epochs", initial=epoch_range[0], total=epoch_range[-1]):
            tstart_epoch = time.time()

            self.train_state = train_epoch(self.train_state, self.partition_fn, self.loss_fn, self.optimizer, train_dl, logger)

            tend_epoch = time.time()

            logger.log(
                {
                    "train/epoch": epoch,
                    "train/epoch_time": tend_epoch - tstart_epoch,
                }
            )

            if val_dl and epoch % epochs_per_val == 0:
                tstart_val = time.time()
                eval_results = self.run_evals(val_dl, eval_suite)
                tend_val = time.time()

                val_loss = np.asarray(eval_results["loss"]).item()
                val_loss_history = np.append(val_loss_history, val_loss)

                # Check for early stopping.
                # If the validation loss has not decreased for the last `patience` number of evaluations, stop training.
                if patience is not None and len(val_loss_history) > patience and np.all(np.diff(val_loss_history[-patience:]) >= 0.0):
                    break

                # Pre-pend "val/" to the keys in the eval_results dictionary.
                eval_results = {f"val/{k}": v for k, v in eval_results.items()}

                logger.log(
                    {
                        "val/epoch": epoch,
                        "val/evaluation_time": tend_val - tstart_val,
                    }
                    | eval_results
                )

                # TODO(allenw): add checkpointing.

    def run_evals(self, dataloader: DataLoader, eval_suite: EvaluationSuite) -> dict[str, typing.Any]:
        eval_results = eval_module_on_data(self.train_state.model, dataloader, eval_suite)
        return eval_results

    def compute_loss(self, dataloader: DataLoader) -> float:
        loss_eval_fn = make_val_loss_eval_fn(self.loss_fn)
        eval_suite = {"loss": loss_eval_fn}
        results = eval_module_on_data(self.train_state.model, dataloader, eval_suite)
        return results["loss"]
