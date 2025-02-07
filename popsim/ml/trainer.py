import time
import typing
import warnings
from os import PathLike

import equinox as eqx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PyTree
from tqdm import tqdm

from popsim.ml._types import TrainableModel
from popsim.ml.checkpointing import TrainState, create_default_checkpoint_manager, restore_train_state, save_train_state
from popsim.ml.dataloading import DataLoader
from popsim.ml.envs import ModuleTrainingEnv
from popsim.ml.eval import EvalData, EvaluationSuite, batch_loss, make_val_loss_eval_fn, run_evals
from popsim.ml.loggers import ConsoleLogger, LoggerBase
from popsim.ml.loss import InstantaneousLoss, IntegralLoss, LossFunction
from popsim.ml.partition import PartitionFn, make_partition_by_members
from popsim.tree_util import any_nans


@eqx.filter_jit
def step_nan_check(model, opt_state, loss_value):
    """Check for NaN values in the model, optimizer state, or loss value."""
    return any_nans((model, opt_state, loss_value))


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
    """Train the model for one step of SGD."""
    # Partition the model into trainable and static parts.
    trainable, static = partition_fn(model)

    def _batch_loss(_trainable: TrainableModel):
        return batch_loss(_trainable, static, loss_fn, inputs, targets)

    # Compute the loss value and the gradient of loss w.r.t. the trainable parts of the model.
    loss_value, grads = eqx.filter_value_and_grad(_batch_loss)(trainable)

    # Update the optimizer and the model.
    model_updates, opt_state = optimizer.update(grads, opt_state, trainable, value=loss_value, grad=grads, value_fn=_batch_loss)

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
) -> TrainState | None:
    """Train the model for one epoch."""
    for batch in train_dl:
        tstart_prep = time.time()
        inputs, targets = batch.get_inputs_and_targets()
        tend_prep = time.time()

        tstart_step = time.time()

        # block_until_ready on JIT compiled functions is important for correctly benchmarking the elapsed time.
        model, opt_state, loss_value = jax.block_until_ready(
            train_step(
                train_state.model,
                partition_fn,
                loss_fn,
                optimizer,
                train_state.opt_state,
                inputs,
                targets,
            )
        )

        if step_nan_check(model, opt_state, loss_value):
            warnings.warn("NaN values found in model, optimizer state, or loss value. Stopping training.", stacklevel=2)
            return None

        tend_step = time.time()

        logger.log(
            {
                "train/loss": loss_value,
                "train/step": train_state.step,
                "train/step_time": tend_step - tstart_step,
                "train/prep_time": tend_prep - tstart_prep,
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
        loss_fn: LossFunction,
        optimizer: optax.GradientTransformation,
        checkpoint_dir: typing.Optional[PathLike] = None,
        trainable_getter: typing.Optional[typing.Callable[[TrainableModel], PyTree]] = None,
        grad_clip: typing.Optional[optax.GradientTransformation] = None,
    ):
        """Initialize a Trainer object.

        Args:
            model (TrainableModel): the model to train.
            loss_fn (LossFunction): the loss function to use.
            optimizer (optax.GradientTransformation): the optimizer to use.
            checkpoint_dir (typing.Optional[PathLike], optional): path to the directory to save checkpoints at / load checkpoints from. Defaults to None.
            trainable_getter (typing.Optional[typing.Callable[[TrainableModel], PyTree]], optional): A function to specify what parameters in the model to train; the rest will be not be trained. This function takes in a model instance and outputs a PyTree (e.g. tuple or list) of parameters to train. Defaults to None.
            grad_clip (typing.Optional[optax.GradientTransformation], optional): Gradient clipping to apply before optimizer to avoid training instability. If None, then will default to optax.clip_by_global_norm(0.5). Defaults to None.

        """
        if isinstance(model, ModuleTrainingEnv):
            assert isinstance(loss_fn, IntegralLoss), "When using a ModuleTrainingEnv, the loss function must be an IntegralLoss."
            partition_fn = make_partition_by_members(lambda m: m.get_trainable())
            if trainable_getter:
                raise ValueError("trainable_getter is not supposed to be provided when training a ModuleTrainingEnv.")
        else:
            partition_fn = make_partition_by_members(trainable_getter or (lambda m: m))

        # Add gradient clipping to the optimizer.
        grad_clip = grad_clip or optax.clip_by_global_norm(0.5)
        optimizer = optax.chain(grad_clip, optimizer)
        self.partition_fn = partition_fn
        self.train_state = TrainState.create_new(model, self.partition_fn, optimizer)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_manager = create_default_checkpoint_manager(checkpoint_dir) if checkpoint_dir else None

    def train(
        self,
        train_dl: DataLoader,
        val_dl: typing.Optional[DataLoader] = None,
        eval_suite: typing.Optional[EvaluationSuite] = None,
        max_epochs: int = 1000,
        epochs_per_val: int = 1,
        logger: typing.Optional[LoggerBase] = None,
    ):
        """Train the model with periodic validation.

        Args:
            train_dl (DataLoader): DataLoader for training the model.
            val_dl (typing.Optional[DataLoader]): DataLoader for validating the model. Defaults to None.
            eval_suite (typing.Optional[EvaluationSuite], optional): Evaluation suite to run periodically. Defaults to None.
            max_epochs (int, optional): Maximum number of epochs to train for. Defaults to 1000.
            epochs_per_val (int, optional): How often to run evaluations. Defaults to 1.
            logger (typing.Optional[LoggerBase], optional): Logger to record results. Defaults to None.
        """
        logger = logger or ConsoleLogger()
        eval_suite = eval_suite or {}
        if "loss" not in eval_suite:
            eval_suite["loss"] = make_val_loss_eval_fn(self.loss_fn)

        if not val_dl and self.checkpoint_manager:
            warnings.warn("No validation DataLoader provided. Checkpoints will not be saved.", stacklevel=2)

        val_loss_history = np.array([])

        # epoch_range accounts for restarting training from a checkpoint.
        epoch_range = range(self.train_state.epoch, self.train_state.epoch + max_epochs + 1)

        for epoch in tqdm(epoch_range, desc="Epochs", initial=epoch_range[0], total=epoch_range[-1]):
            tstart_epoch = time.time()

            new_train_state = train_epoch(self.train_state, self.partition_fn, self.loss_fn, self.optimizer, train_dl, logger)

            if new_train_state is None:
                break
            else:
                self.train_state = new_train_state

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

                # Pre-pend "val/" to the keys in the eval_results dictionary.
                eval_results = {f"val/{k}": v for k, v in eval_results.items()}

                logger.log(
                    {
                        "val/epoch": epoch,
                        "val/evaluation_time": tend_val - tstart_val,
                    }
                    | eval_results
                )

                if self.checkpoint_manager:
                    save_train_state(train_state=self.train_state, checkpoint_manager=self.checkpoint_manager, loss=float(val_loss["mean"]))

    def restore_best_checkpoint(self, path: typing.Optional[PathLike] = None):
        """Restore the best checkpoint. If no path is provided, restore from the path provided to the current checkpoint manager. If a path is provided, restore from the provided path.

        Args:
            path (typing.Optional[PathLike], optional): path to the checkpoint that should be loaded. Defaults to None.

        """
        if path:
            checkpoint_manager = create_default_checkpoint_manager(path)
            self.train_state = restore_train_state(checkpoint_manager, self.train_state)
        else:
            if not self.checkpoint_manager:
                raise ValueError("A path is not provided and the trainer doesn't have a checkpoint manager.")
            self.train_state = restore_train_state(self.checkpoint_manager, self.train_state)

    def run_evals(
        self, dataloader: DataLoader, eval_suite: typing.Optional[EvaluationSuite] = None
    ) -> typing.Union[dict[str, typing.Any], EvalData]:
        """Run an evaluation suite on the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader to evaluate the model on.
            evaluation_suite (Optional[EvaluationSuite], optional): Optional evaluation suite to run. This is a dictionary of evaluation functions that take in an EvalData structure and returns the evaluation results. If None is provided, just return the EvalData generated. Defaults to None.

        Returns:
            typing.Union[dict[str, typing.Any], EvalData]: Dictionary of results from running the evaluation suite.
        """
        eval_results = run_evals(self.train_state.model, dataloader, eval_suite)
        return eval_results

    def compute_loss(self, dataloader: DataLoader) -> dict[str, typing.Any]:
        """Compute the loss of the model on the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader to evaluate the model on.

        Returns:
            dict[str, Any]: Dictionary of results from running the evaluation function generated by make_val_loss_eval_fn.
        """
        loss_eval_fn = make_val_loss_eval_fn(self.loss_fn)
        eval_suite = {"loss": loss_eval_fn}
        results = run_evals(self.train_state.model, dataloader, eval_suite)
        return results["loss"]
