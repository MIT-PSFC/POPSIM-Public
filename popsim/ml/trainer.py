import time
import typing
import warnings
from os import PathLike

import equinox as eqx
import jax
import jax.numpy as jnp
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
    restore_train_state,
    save_train_state,
)
from popsim.ml.dataloading import DataLoader, XarrayPreppedDataset
from popsim.ml.debug_utils import diagnose_nans
from popsim.ml.envs import ModuleTrainingEnv
from popsim.ml.eval import (
    EvalData,
    EvaluationSuite,
    make_val_loss_eval_fn,
    masked_batch_loss,
    run_evals,
)
from popsim.ml.loggers import ConsoleLogger, LoggerBase
from popsim.ml.loss import IntegralLoss, LossFunction
from popsim.ml.partition import PartitionFn, make_partition_by_members
from popsim.tree_util import any_nans


@eqx.filter_jit
def step_nan_check(model, opt_state, loss_value):
    """Check for NaN values in the model, optimizer state, or loss value."""
    return any_nans((model, opt_state, loss_value))


# NaN-recovery policy for train steps that produce non-finite values.
#   MAX_NAN_STEP_WARNINGS_PER_EPOCH: per-step NaN warnings logged before suppressing the
#     rest of the epoch (the epoch summary still reports totals and culprit sample ids).
#   MAX_LOGGED_BAD_SAMPLES: maximum number of culprit sample ids listed in one log line.
#   SKIP_FRACTION_ABORT_THRESHOLD / MAX_HIGH_SKIP_EPOCHS: abort training when more than
#     this fraction of steps is skipped for this many consecutive epochs, since the model
#     is then training on a small biased subset of the data and cannot make reliable progress.
MAX_NAN_STEP_WARNINGS_PER_EPOCH = 3
MAX_LOGGED_BAD_SAMPLES = 50
SKIP_FRACTION_ABORT_THRESHOLD = 0.5
MAX_HIGH_SKIP_EPOCHS = 3


class TrainingDivergedError(RuntimeError):
    """Train steps keep producing NaN/Inf and masking the bad samples cannot recover them.

    Distinct from data errors (NaN inputs or targets).
    Trainer.train catches it and falls back to the best checkpoint for the test eval,
    or re-raises when no best checkpoint exists yet.
    """


def _format_sample_ids(sample_ids) -> str:
    """Format a collection of sample coordinate values for logging, truncated to MAX_LOGGED_BAD_SAMPLES."""
    ids = sorted(sample_ids, key=str)
    formatted = ", ".join(str(v) for v in ids[:MAX_LOGGED_BAD_SAMPLES])
    if len(ids) > MAX_LOGGED_BAD_SAMPLES:
        formatted += f", ... ({len(ids) - MAX_LOGGED_BAD_SAMPLES} more)"
    return formatted


def _record_bad_samples(batch: XarrayPreppedDataset, mask_bad: np.ndarray, bad_sample_ids: set) -> list:
    """Add the sample coordinate values where mask_bad is True to bad_sample_ids and return them."""
    sample_ids = batch.sample_coord.values
    new_bad = list(sample_ids[mask_bad])
    bad_sample_ids.update(new_bad)
    return new_bad


def _high_skip_epoch_count(train_metrics: dict | None, consecutive_high_skip_epochs: int, epoch: int) -> int:
    """Return the updated count of consecutive epochs that skipped most steps due to NaN/Inf.

    A single bad epoch can recover because batches are reshuffled,
    only stop training if there is a prolonged streak of NaN steps.
    Raises TrainingDivergedError once the streak reaches MAX_HIGH_SKIP_EPOCHS.
    """
    skip_fraction = (train_metrics or {}).get("train/nan_skip_fraction", 0.0)
    if skip_fraction <= SKIP_FRACTION_ABORT_THRESHOLD:
        return 0
    consecutive_high_skip_epochs += 1
    if consecutive_high_skip_epochs >= MAX_HIGH_SKIP_EPOCHS:
        raise TrainingDivergedError(
            f"More than {SKIP_FRACTION_ABORT_THRESHOLD:.0%} of train steps skipped due to NaN/Inf "
            f"for {consecutive_high_skip_epochs} consecutive epochs "
            f"(epoch {epoch}: {skip_fraction:.0%} skipped). Training cannot make reliable progress."
        )
    return consecutive_high_skip_epochs


@eqx.filter_jit
def train_step(
    model: TrainableModel,
    partition_fn: PartitionFn,
    loss_fn: LossFunction,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    inputs: PyTree[Array],
    targets: PyTree[Array],
    sample_mask: Array,
) -> tuple[TrainableModel, optax.OptState, Array, Array]:
    """Train the model for one step of SGD on the mean loss over the samples where sample_mask is True.

    sample_mask is always a boolean array over the batch, so padded and NaN-retry steps reuse one compilation.
    Returns the new model, the new optimizer state, the masked mean loss, and the per-sample losses.
    """
    # Partition the model into trainable and static parts.
    trainable, static = partition_fn(model)

    def _masked_loss_and_sample_losses(_trainable: TrainableModel):
        return masked_batch_loss(_trainable, static, loss_fn, inputs, targets, sample_mask)

    def _masked_loss(_trainable: TrainableModel):
        masked_mean_loss, _ = _masked_loss_and_sample_losses(_trainable)
        return masked_mean_loss

    # Compute the loss value and the gradient of loss w.r.t. the trainable parts of the model.
    loss_and_sample_losses, grads = eqx.filter_value_and_grad(_masked_loss_and_sample_losses, has_aux=True)(trainable)
    loss_value, sample_losses = loss_and_sample_losses

    # Line-search optimizers such as lbfgs need a scalar value_fn.
    model_updates, opt_state = optimizer.update(grads, opt_state, trainable, value=loss_value, grad=grads, value_fn=_masked_loss)

    # Apply the updates to the trainable part of the model.
    trainable = eqx.apply_updates(trainable, model_updates)

    # Combine the trainable and static parts of the model to get back the whole model.
    new_model = eqx.combine(trainable, static)
    return new_model, opt_state, loss_value, sample_losses


def _retry_step_without_nonfinite_samples(
    train_state: TrainState,
    partition_fn: PartitionFn,
    loss_fn: LossFunction,
    optimizer: optax.GradientTransformation,
    batch: XarrayPreppedDataset,
    sample_losses: Array,
    mask_valid: np.ndarray,
    bad_sample_ids: set,
    verbose: bool,
) -> tuple[TrainableModel, optax.OptState, Array] | None:
    """Retake a NaN/Inf train step with the samples whose loss is non-finite masked out.

    Non-finite samples are overwritten with a finite sample so the backward pass stays finite,
    then masked out so they add no loss or gradient.
    The retry has the same shapes and dtypes, so it reuses the compiled train_step.
    Returns the (model, opt_state, loss_value) of the retried step, or None when the step must be skipped.
    The step is skipped when every per-sample loss is finite, since the NaN/Inf then came from the backward pass or optimizer update.
    It is also skipped when no valid sample is finite, or when the retry still produces NaN/Inf.
    """
    step, epoch = train_state.step, train_state.epoch
    mask_nonfinite = ~np.isfinite(np.asarray(sample_losses))
    mask_bad = mask_valid & mask_nonfinite
    mask_good = mask_valid & ~mask_nonfinite
    new_bad = _record_bad_samples(batch, mask_bad, bad_sample_ids)

    if not mask_nonfinite.any():
        if verbose:
            loguru.logger.warning(
                "NaN/Inf after train_step at step {} (epoch {}) but all per-sample losses are finite, "
                "so the non-finite values arose in the backward pass or optimizer update and the culprit samples "
                "cannot be isolated. Skipping optimizer update.",
                step,
                epoch,
            )
        return None
    if not mask_good.any():
        if verbose:
            loguru.logger.warning(
                "NaN/Inf after train_step at step {} (epoch {}) and all {} samples have a non-finite loss. Skipping optimizer update.",
                step,
                epoch,
                int(mask_valid.sum()),
            )
        return None

    if verbose:
        loguru.logger.warning(
            "NaN/Inf after train_step at step {} (epoch {}). Retrying with {}/{} non-finite samples masked out: [{}]",
            step,
            epoch,
            int(mask_bad.sum()),
            int(mask_valid.sum()),
            _format_sample_ids(new_bad),
        )
    # Padded duplicates of a bad sample are non-finite too, so replace every non-finite row.
    # Re-indexing re-reads lazy zarr or dask data from disk, should be fine since this only runs on NaN steps.
    first_good_idx = np.argmax(mask_good)
    all_row_idx = np.arange(mask_nonfinite.size)
    row_idx_finite = np.where(mask_nonfinite, first_good_idx, all_row_idx)
    batch_finite = batch[row_idx_finite]
    inputs_finite, targets_finite = batch_finite.get_inputs_and_targets()
    sample_mask_good = jnp.asarray(mask_good)
    step_outputs = train_step(
        train_state.model,
        partition_fn,
        loss_fn,
        optimizer,
        train_state.opt_state,
        inputs_finite,
        targets_finite,
        sample_mask_good,
    )
    model, opt_state, loss_value, _ = jax.block_until_ready(step_outputs)
    if step_nan_check(model, opt_state, loss_value):
        if verbose:
            loguru.logger.warning(
                "Masked retry at step {} (epoch {}) still produced NaN/Inf, skipping optimizer update.",
                step,
                epoch,
            )
        return None
    return model, opt_state, loss_value


def train_epoch(
    train_state: TrainState,
    partition_fn: PartitionFn,
    loss_fn: LossFunction,
    optimizer: optax.GradientTransformation,
    train_dl: DataLoader,
) -> tuple[TrainState, dict[str, typing.Any] | None]:
    """Train the model for one epoch. Returns the new train state and the metrics of the last step."""
    train_metrics = None
    n_applied = 0
    n_masked = 0
    n_skipped = 0
    n_nan_events = 0
    bad_sample_ids: set = set()
    for batch in train_dl:
        tstart_prep = time.time()
        inputs, targets = batch.get_inputs_and_targets()
        if any_nans(inputs):
            raise RuntimeError("NaN values found in inputs. Training cannot continue.")
        if any_nans(targets):
            raise RuntimeError("NaN values found in targets, this will cause further NaNs on backpropagation. Training cannot continue.")
        # Padded rows repeat the last real sample, mask them so they add no extra weight.
        n_batch = len(batch)
        mask_valid = np.arange(n_batch) < n_batch - batch.n_padded
        sample_mask_valid = jnp.asarray(mask_valid)
        tend_prep = time.time()

        tstart_step = time.time()

        # block_until_ready on JIT compiled functions is important for correctly benchmarking the elapsed time.
        step_outputs = train_step(
            train_state.model,
            partition_fn,
            loss_fn,
            optimizer,
            train_state.opt_state,
            inputs,
            targets,
            sample_mask_valid,
        )
        model, opt_state, loss_value, sample_losses = jax.block_until_ready(step_outputs)

        if step_nan_check(model, opt_state, loss_value):
            # Some samples can drive the solve to a non-finite loss, gradient or update.
            # Retry without them so the rest of the batch still trains.
            # Otherwise skip the step and keep the last good params and optimizer state.
            # Per-step warnings are capped, the epoch summary below reports totals and culprits.
            n_nan_events += 1
            recovered = _retry_step_without_nonfinite_samples(
                train_state,
                partition_fn,
                loss_fn,
                optimizer,
                batch,
                sample_losses,
                mask_valid,
                bad_sample_ids,
                verbose=n_nan_events <= MAX_NAN_STEP_WARNINGS_PER_EPOCH,
            )
            if recovered is None:
                n_skipped += 1
                continue
            model, opt_state, loss_value = recovered
            n_masked += 1

        tend_step = time.time()
        n_applied += 1

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

    n_total = n_applied + n_skipped
    if n_nan_events:
        loguru.logger.warning(
            "Epoch {}: {}/{} steps hit NaN/Inf ({} recovered by masking, {} skipped). Non-finite samples seen: [{}]",
            train_state.epoch,
            n_nan_events,
            n_total,
            n_masked,
            n_skipped,
            _format_sample_ids(bad_sample_ids),
        )
    if n_applied == 0 and n_skipped > 0:
        # Every step this epoch diverged and none could be recovered by masking: the model
        # cannot make progress, so surface the failure with diagnostics rather than looping
        # forever on skipped steps.
        diagnose_nans(train_state.model, batch)
        raise TrainingDivergedError(
            f"All {n_skipped} steps this epoch produced NaN/Inf after train_step and could not be recovered "
            "by masking, training cannot make progress."
        )
    if train_metrics is not None:
        # Report NaN-recovery stats so Trainer.train can abort runs that skip most steps.
        train_metrics["train/nan_steps_masked"] = n_masked
        train_metrics["train/nan_steps_skipped"] = n_skipped
        train_metrics["train/nan_skip_fraction"] = n_skipped / max(n_total, 1)

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
        checkpoint_max_to_keep: int = 1,
    ):
        """Initialize a Trainer object.

        Args:
            model (TrainableModel): the model to train.
            loss_fn (LossFunction): the loss function to use.
            optimizer (optax.GradientTransformation): the optimizer to use.
            checkpoint_dir (typing.Optional[PathLike], optional): path to the directory to save checkpoints at / load checkpoints from. Defaults to None.
            trainable_getter (typing.Optional[typing.Callable[[TrainableModel], PyTree]], optional): A function to specify what parameters in the model to train; the rest will be not be trained. This function takes in a model instance and outputs a PyTree (e.g. tuple or list) of parameters to train. Defaults to None.
            grad_clip (typing.Optional[optax.GradientTransformation], optional): Gradient clipping to apply before optimizer to avoid training instability. If None, then will default to optax.clip_by_global_norm(0.5). Defaults to None.
            resume (bool, optional): If True and checkpoint_dir has a checkpoint, restore the latest one (model, optimizer state, and epoch counter) and continue training from there. Defaults to False.
            checkpoint_max_to_keep (int, optional): How many best-by-validation-loss checkpoints to keep. The latest checkpoint is always kept as well. Defaults to 1.

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
        self.checkpoint_manager = (
            create_default_checkpoint_manager(checkpoint_dir, max_to_keep=checkpoint_max_to_keep) if checkpoint_dir else None
        )

        latest_step = self.checkpoint_manager.latest_step() if (resume and self.checkpoint_manager) else None
        if latest_step is not None:
            self.train_state = restore_train_state(self.checkpoint_manager, self.train_state, step=latest_step)
            loguru.logger.info(
                f"Resumed train state from latest checkpoint at epoch {self.train_state.epoch} (step {self.train_state.step})"
            )

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

        If training diverges (TrainingDivergedError), the test eval runs on the best checkpoint instead.
        The error is re-raised when no best checkpoint exists yet.

        Args:
            train_dl (DataLoader): DataLoader for training the model.
            val_dl (typing.Optional[DataLoader]): DataLoader for validating the model. Defaults to None.
            eval_suite (typing.Optional[EvaluationSuite], optional): Evaluation suite to run periodically. Defaults to None.
            max_epochs (int, optional): Total number of epochs to train. A run resumed after N completed epochs trains the remaining max_epochs - N (absolute target, not additive). Defaults to 1000.
            epochs_per_val (int, optional): Validate after every epochs_per_val completed epochs, and always after the final epoch. Defaults to 1.
            patience (int | None, optional): Number of validation steps with no improvement in validation loss before stopping early. Defaults to None (no early stopping). Note the patience counter is not persisted across resumed runs.
            logger (typing.Optional[LoggerBase], optional): Logger to record results. When logger.stop_requested() is True, training stops at the next epoch boundary and returns None without the test eval. Defaults to None.
            max_wall_seconds (float | None, optional): Wall-clock budget for this call. When exceeded, save the latest checkpoint and stop WITHOUT running the test eval, returning None, so a later job can resume and finish. Defaults to None (no budget).
        """
        logger = logger or ConsoleLogger()
        eval_suite = eval_suite or {}
        if "loss" not in eval_suite:
            eval_suite["loss"] = make_val_loss_eval_fn(self.loss_fn)

        if not val_dl and self.checkpoint_manager:
            warnings.warn("No validation DataLoader provided. Best checkpoints will not be saved.", stacklevel=2)

        # Log summary metrics of the dataloaders.
        logger.log({"train_dl": train_dl.metrics, "val_dl": val_dl.metrics if val_dl else None})

        early_stopping = EarlyStopping(patience) if patience is not None else None
        interrupted = False
        consecutive_high_skip_epochs = 0
        tstart_train = time.time()

        # max_epochs is an absolute target, a run resumed after N completed epochs trains the remaining max_epochs - N.
        # An empty range (a resumed run that finished training but died before the test eval) goes straight to the test eval.
        start_epoch = self.train_state.epoch
        if start_epoch > 0:
            loguru.logger.info(f"Resuming training after {start_epoch} of {max_epochs} epochs")
        epoch_range = range(start_epoch, max_epochs)

        for epoch in tqdm(epoch_range, desc="Epochs", initial=start_epoch, total=max_epochs):
            tstart_epoch = time.time()

            try:
                new_train_state, train_metrics = train_epoch(self.train_state, self.partition_fn, self.loss_fn, self.optimizer, train_dl)
                self.train_state = new_train_state
                consecutive_high_skip_epochs = _high_skip_epoch_count(train_metrics, consecutive_high_skip_epochs, epoch)
            except TrainingDivergedError as exc:
                if self.checkpoint_manager is None or self.checkpoint_manager.best_step() is None:
                    raise
                loguru.logger.warning(f"Training diverged at epoch {epoch}, falling back to the best checkpoint. {exc}")
                break

            tend_epoch = time.time()

            # Epoch metrics and checkpoints count completed epochs, so the final epoch is max_epochs.
            n_epochs_completed = self.train_state.epoch
            is_val_epoch = n_epochs_completed % epochs_per_val == 0 or n_epochs_completed == max_epochs
            self._log_epoch_metrics(logger, n_epochs_completed, is_val_epoch, train_metrics, tend_epoch - tstart_epoch)

            if val_dl and is_val_epoch:
                tstart_val = time.time()
                eval_results = self.run_evals(val_dl, eval_suite)
                tend_val = time.time()

                val_loss_mean = float(eval_results["loss"]["mean"])

                # Pre-pend "val/" to the keys in the eval_results dictionary.
                eval_results = {f"val/{k}": v for k, v in eval_results.items()}

                logger.log(
                    {
                        "val/epoch": n_epochs_completed,
                        "val/evaluation_time": tend_val - tstart_val,
                    }
                    | eval_results
                )

                self._save_checkpoint(val_loss_mean)

                if early_stopping is not None and early_stopping.should_stop(val_loss_mean):
                    loguru.logger.info(
                        f"Early stopping: validation loss has not improved for {patience} validation steps ({patience * epochs_per_val} epochs)."
                    )
                    break

            if self._wall_budget_exceeded(tstart_train, max_wall_seconds):
                interrupted = True
                break

            if logger.stop_requested():
                loguru.logger.info(f"Stop requested at epoch {self.train_state.epoch}, stopping without the test eval.")
                interrupted = True
                break

        if interrupted:
            return None
        return self._best_checkpoint_test_results(test_dl, test_eval_suite, logger)

    def _best_checkpoint_test_results(
        self, test_dl: DataLoader | None, test_eval_suite: EvaluationSuite | None, logger: LoggerBase
    ) -> dict[str, typing.Any] | None:
        """Restore the best checkpoint and return its test results.
        Returns None if a checkpoint manager or a test set was not specified.
        """
        if self.checkpoint_manager is None or test_dl is None or test_eval_suite is None:
            loguru.logger.info("No checkpoint manager or test DataLoader provided. Skipping test evaluation.")
            return None
        loguru.logger.info("Restoring the best checkpoint and evaluating on the test set.")
        self.restore_best_checkpoint()
        test_results = self.run_evals(test_dl, eval_suite=test_eval_suite)
        test_results = {f"test/{k}": v for k, v in test_results.items()}
        logger.log(test_results)
        return test_results

    @staticmethod
    def _log_epoch_metrics(logger: LoggerBase, n_epochs_completed: int, is_val_epoch: bool, train_metrics: dict | None, epoch_time: float):
        """Log the epoch count and time together with the metrics of the epoch's last step."""
        if logger.log_at_val_cadence and not is_val_epoch:
            return
        epoch_train_metrics = {
            "train/epoch": n_epochs_completed,
            "train/epoch_time": epoch_time,
        }
        if train_metrics is not None:
            epoch_train_metrics = epoch_train_metrics | train_metrics
        logger.log(epoch_train_metrics)

    def _save_checkpoint(self, val_loss_mean: float | None):
        """Save the current train state, ranked by val_loss_mean, or kept only as the latest when it is None."""
        if self.checkpoint_manager:
            save_train_state(train_state=self.train_state, checkpoint_manager=self.checkpoint_manager, loss=val_loss_mean)

    def _wall_budget_exceeded(self, tstart_train: float, max_wall_seconds: float | None) -> bool:
        """Check the wall-clock budget, saving the latest checkpoint before reporting it exceeded."""
        if max_wall_seconds is None or (time.time() - tstart_train) <= max_wall_seconds:
            return False
        # Save progress since the last validation (does nothing when this epoch was just validated and saved)
        self._save_checkpoint(val_loss_mean=None)
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
