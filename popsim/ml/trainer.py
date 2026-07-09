import time
import typing
import warnings
from os import PathLike

import equinox as eqx
import jax
import loguru
import numpy as np
import optax
import orbax.checkpoint as ocp
from jaxtyping import Array, PyTree
from tqdm import tqdm

from popsim.ml._types import TrainableModel
from popsim.ml.checkpointing import (
    TrainState,
    create_default_checkpoint_manager,
    create_latest_checkpoint_manager,
    latest_checkpoint_dir,
    restore_train_state,
    save_train_state,
)
from popsim.ml.dataloading import DataLoader
from popsim.ml.debug_utils import diagnose_nans
from popsim.ml.envs import ModuleTrainingEnv
from popsim.ml.eval import EvalData, EvaluationSuite, batch_loss, make_val_loss_eval_fn, run_evals
from popsim.ml.loggers import ConsoleLogger, LoggerBase, WandbLogger
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
) -> tuple[TrainState, dict[str, typing.Any] | None]:
    """Train the model for one epoch. Returns the new train state and the metrics of the last step."""
    train_metrics = None
    for batch in train_dl:
        tstart_prep = time.time()
        inputs, targets = batch.get_inputs_and_targets()
        if any_nans(inputs):
            raise RuntimeError("NaN values found in inputs. Training cannot continue.")
        if any_nans(targets):
            raise RuntimeError("NaN values found in targets, this will cause further NaNs on backpropagation. Training cannot continue.")
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
            diagnose_nans(train_state.model, batch)
            raise RuntimeError("NaN values found in model, optimizer state, or loss value after train_step. Training cannot continue.")

        tend_step = time.time()

        train_state = TrainState(
            step=train_state.step + 1,
            epoch=train_state.epoch,
            model=model,
            opt_state=opt_state,
        )
        train_metrics = {
            "train/loss": loss_value,
            "train/step": train_state.step,
            "train/step_time": tend_step - tstart_step,
            "train/prep_time": tend_prep - tstart_prep,
        }

    # Only log the last step's values
    # Logging every batch overloads the W&B backend for fast-training models.
    # For wandb, train_metrics are instead logged by Trainer.train at the validation cadence.
    if (not isinstance(logger, WandbLogger)) and train_metrics is not None:
        logger.log(train_metrics)
    train_state.epoch += 1
    return train_state, train_metrics


class EarlyStopping:
    """Track the validation loss and signal when it has stopped improving for `patience` checks."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


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
        checkpoint_dir: PathLike | None = None,
        trainable_getter: typing.Callable[[TrainableModel], PyTree] | None = None,
        grad_clip: optax.GradientTransformation | None = None,
        resume: bool = False,
    ):
        """Initialize a Trainer object.

        Args:
            model (TrainableModel): the model to train.
            loss_fn (LossFunction): the loss function to use.
            optimizer (optax.GradientTransformation): the optimizer to use.
            checkpoint_dir (typing.Optional[PathLike], optional): path to the directory to save checkpoints at / load checkpoints from. Defaults to None.
            trainable_getter (typing.Optional[typing.Callable[[TrainableModel], PyTree]], optional): A function to specify what parameters in the model to train; the rest will be not be trained. This function takes in a model instance and outputs a PyTree (e.g. tuple or list) of parameters to train. Defaults to None.
            grad_clip (typing.Optional[optax.GradientTransformation], optional): Gradient clipping to apply before optimizer to avoid training instability. If None, then will default to optax.clip_by_global_norm(0.5). Defaults to None.
            resume (bool, optional): If True and a latest checkpoint exists in <checkpoint_dir>_latest, restore it (model, optimizer state, and epoch counter) and continue training from there. Defaults to False.

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
        # Created lazily so restore-only Trainers (e.g. results collection) do not
        # leave empty <checkpoint_dir>_latest directories behind
        self._latest_checkpoint_dir = latest_checkpoint_dir(checkpoint_dir) if checkpoint_dir else None
        self._latest_checkpoint_manager: ocp.CheckpointManager | None = None

        if resume and self.latest_checkpoint_manager and self.latest_checkpoint_manager.latest_step() is not None:
            self.train_state = restore_train_state(self.latest_checkpoint_manager, self.train_state)
            loguru.logger.info(
                f"Resumed train state from latest checkpoint at epoch {self.train_state.epoch} (step {self.train_state.step})"
            )

    @property
    def latest_checkpoint_manager(self) -> ocp.CheckpointManager | None:
        """CheckpointManager keeping the most recent checkpoint, used for resuming interrupted runs."""
        if self._latest_checkpoint_manager is None and self._latest_checkpoint_dir is not None:
            self._latest_checkpoint_manager = create_latest_checkpoint_manager(self._latest_checkpoint_dir)
        return self._latest_checkpoint_manager

    def train(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader | None = None,
        test_dl: DataLoader | None = None,
        eval_suite: EvaluationSuite | None = None,
        test_eval_suite: EvaluationSuite | None = None,
        max_epochs: int = 1000,
        epochs_per_val: int = 1,
        patience: int | None = None,
        logger: LoggerBase | None = None,
        max_wall_seconds: float | None = None,
    ):
        """Train the model with periodic validation.

        Args:
            train_dl (DataLoader): DataLoader for training the model.
            val_dl (typing.Optional[DataLoader]): DataLoader for validating the model. Defaults to None.
            eval_suite (typing.Optional[EvaluationSuite], optional): Evaluation suite to run periodically. Defaults to None.
            max_epochs (int, optional): Total epoch budget. When resuming from a checkpoint at epoch N, training continues from N up to max_epochs (absolute target, not additive). Defaults to 1000.
            epochs_per_val (int, optional): How often to run evaluations. Defaults to 1.
            patience (int | None, optional): Number of validation steps with no improvement in validation loss before stopping early. Defaults to None (no early stopping). Note the patience counter is not persisted across resumed runs.
            logger (typing.Optional[LoggerBase], optional): Logger to record results. Defaults to None.
            max_wall_seconds (float | None, optional): Wall-clock budget for this call. When exceeded, save the latest checkpoint and stop WITHOUT running the test eval, returning None, so a later job can resume and finish. Defaults to None (no budget).
        """
        logger = logger or ConsoleLogger()
        eval_suite = eval_suite or {}
        if "loss" not in eval_suite:
            eval_suite["loss"] = make_val_loss_eval_fn(self.loss_fn)

        if not val_dl and self.checkpoint_manager:
            warnings.warn("No validation DataLoader provided. Checkpoints will not be saved.", stacklevel=2)

        # Log summary metrics of the dataloaders.
        logger.log({"train_dl": train_dl.metrics, "val_dl": val_dl.metrics if val_dl else None})

        early_stopping = EarlyStopping(patience) if patience is not None else None
        timed_out = False
        tstart_train = time.time()

        # max_epochs is an absolute target: a run resumed at epoch N trains N..max_epochs.
        # An empty range (resumed run that already finished training but died before
        # the test eval) skips straight to restoring the best checkpoint and evaluating.
        start_epoch = self.train_state.epoch
        if start_epoch > 0:
            loguru.logger.info(f"Resuming training at epoch {start_epoch} of {max_epochs}")
        epoch_range = range(start_epoch, max_epochs + 1)

        for epoch in tqdm(epoch_range, desc="Epochs", initial=start_epoch, total=max_epochs):
            tstart_epoch = time.time()

            new_train_state, train_metrics = train_epoch(
                self.train_state, self.partition_fn, self.loss_fn, self.optimizer, train_dl, logger
            )
            self.train_state = new_train_state

            tend_epoch = time.time()

            # For fast-training models, W&B can't handle logging on every epoch, so log at the validation cadence.
            if (not isinstance(logger, WandbLogger)) or (epoch % epochs_per_val == 0):
                epoch_train_metrics = {
                    "train/epoch": epoch,
                    "train/epoch_time": tend_epoch - tstart_epoch,
                }
                # For W&B, include the latest train-step metrics at the same cadence as validation logs.
                if isinstance(logger, WandbLogger) and train_metrics is not None:
                    epoch_train_metrics = epoch_train_metrics | train_metrics
                logger.log(epoch_train_metrics)

            if val_dl and epoch % epochs_per_val == 0:
                tstart_val = time.time()
                eval_results = self.run_evals(val_dl, eval_suite)
                tend_val = time.time()

                val_loss = np.asarray(eval_results["loss"]).item()
                val_loss_mean = float(val_loss["mean"])

                # Pre-pend "val/" to the keys in the eval_results dictionary.
                eval_results = {f"val/{k}": v for k, v in eval_results.items()}

                logger.log(
                    {
                        "val/epoch": epoch,
                        "val/evaluation_time": tend_val - tstart_val,
                    }
                    | eval_results
                )

                self._save_checkpoints(val_loss_mean)

                if early_stopping is not None and early_stopping.should_stop(val_loss_mean):
                    loguru.logger.info(
                        f"Early stopping: validation loss has not improved for {patience} validation steps ({patience * epochs_per_val} epochs)."
                    )
                    break

            if self._wall_budget_exceeded(tstart_train, max_wall_seconds, early_stopping):
                timed_out = True
                break

        if timed_out:
            return None

        if self.checkpoint_manager is None or test_dl is None or test_eval_suite is None:
            loguru.logger.info("No checkpoint manager or test DataLoader provided. Skipping test evaluation.")
            return None
        else:
            loguru.logger.info("Training completed. Restoring the best checkpoint and evaluating on the test set.")
            self.restore_best_checkpoint()
            test_results = self.run_evals(test_dl, eval_suite=test_eval_suite)
            test_results = {f"test/{k}": v for k, v in test_results.items()}
            logger.log(test_results)
            return test_results

    def _save_checkpoints(self, val_loss_mean: float):
        """Save the best-by-val-loss checkpoint and the latest (resume) checkpoint."""
        if self.checkpoint_manager:
            save_train_state(train_state=self.train_state, checkpoint_manager=self.checkpoint_manager, loss=val_loss_mean)
        if self.latest_checkpoint_manager:
            save_train_state(train_state=self.train_state, checkpoint_manager=self.latest_checkpoint_manager, loss=val_loss_mean)

    def _wall_budget_exceeded(self, tstart_train: float, max_wall_seconds: float | None, early_stopping: "EarlyStopping | None") -> bool:
        """Check the wall-clock budget, saving the latest checkpoint before reporting it exceeded."""
        if max_wall_seconds is None or (time.time() - tstart_train) <= max_wall_seconds:
            return False
        if self.latest_checkpoint_manager and self.latest_checkpoint_manager.latest_step() != self.train_state.epoch:
            # Save progress made since the last validation checkpoint. The loss
            # metric is informational only, this manager keeps the latest step.
            save_train_state(
                train_state=self.train_state,
                checkpoint_manager=self.latest_checkpoint_manager,
                loss=early_stopping.best_loss if early_stopping is not None else float("inf"),
            )
        loguru.logger.info(
            f"Wall-clock budget of {max_wall_seconds} s exceeded at epoch {self.train_state.epoch}. "
            "Saved the latest checkpoint and stopping without the test eval so a resumed job can finish training."
        )
        return True

    def restore_best_checkpoint(self, path: PathLike | None = None):
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

    def run_evals(self, dataloader: DataLoader, eval_suite: EvaluationSuite | None = None) -> dict[str, typing.Any] | EvalData:
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
