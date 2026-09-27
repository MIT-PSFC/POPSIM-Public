"""
Utilities for saving and loading model and training checkpoints.
"""

import os
from os import PathLike

import chex
import equinox as eqx
import optax
import orbax.checkpoint as ocp
from jaxtyping import PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.partition import PartitionFn


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


def _validation_loss(metrics: dict) -> float:
    """The metric checkpoints are ranked by, lower is better."""
    return metrics["loss"]


def create_default_checkpoint_manager(directory: PathLike, max_to_keep: int = 1) -> ocp.CheckpointManager:
    """Create the default CheckpointManager.

    It keeps the max_to_keep best checkpoints by validation loss, plus the latest checkpoint for resuming.
    Nothing is deleted while at most max_to_keep checkpoints exist.
    A checkpoint saved without a loss never counts as best,
    and is deleted once it is no longer the latest and more than max_to_keep checkpoints exist.
    Deletion only happens on save(), so a restore-only manager never deletes checkpoints.

    Args:
        directory (PathLike): the directory to save the checkpoints in.
        max_to_keep (int): how many best-by-validation-loss checkpoints to keep.

    Returns:
        ocp.CheckpointManager: the CheckpointManager.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    policy_latest = ocp.checkpoint_managers.LatestN(n=1)
    policy_best = ocp.checkpoint_managers.BestN(
        get_metric_fn=_validation_loss,
        reverse=True,
        n=max_to_keep,
        keep_checkpoints_without_metrics=False,
    )
    preservation_policy = ocp.checkpoint_managers.AnyPreservationPolicy([policy_latest, policy_best])

    # preservation_policy replaces max_to_keep, orbax raises if both are set.
    # best_fn and best_mode still rank checkpoints for best_step().
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        best_fn=_validation_loss,
        best_mode="min",
        preservation_policy=preservation_policy,
    )

    manager = ocp.CheckpointManager(
        directory=directory,
        options=options,
        item_names=("model", "opt_state", "metadata"),
    )
    return manager


def partition_saveable(pytree: PyTree) -> tuple[PyTree, PyTree]:
    """Partition a pytree into saveable and non-saveable parts."""
    return eqx.partition(pytree, eqx.is_array_like)


def save_train_state(train_state: TrainState, checkpoint_manager: ocp.CheckpointManager, loss: float | None):
    """Save the training state to a checkpoint, a no-op if this epoch was already saved.

    Args:
        train_state (TrainState): the training state to save.
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager to save to.
        loss (float | None): the validation loss of the model, None for an epoch without validation.
    """
    metadata = {
        "step": train_state.step,
        "epoch": train_state.epoch,
    }

    model_save, _ = partition_saveable(train_state.model)
    opt_state_save, _ = partition_saveable(train_state.opt_state)

    composite_save = ocp.args.Composite(
        model=ocp.args.StandardSave(model_save),
        opt_state=ocp.args.StandardSave(opt_state_save),
        metadata=ocp.args.JsonSave(metadata),
    )

    # A None loss must be saved as no metrics, a None metric breaks the best-checkpoint ranking.
    metrics = {"loss": loss} if loss is not None else None
    checkpoint_manager.save(
        train_state.epoch,
        args=composite_save,
        metrics=metrics,
    )
    checkpoint_manager.wait_until_finished()


def _step_to_restore(checkpoint_manager: ocp.CheckpointManager, step: int | None) -> int:
    """The requested step, or the best step by validation loss when step is None.

    Raises instead of passing None to orbax, which would silently restore the latest step.
    """
    step_to_restore = step if step is not None else checkpoint_manager.best_step()
    if step_to_restore is None:
        raise FileNotFoundError(
            f"No validated checkpoint in {checkpoint_manager.directory}. "
            "Checkpoints saved without a validation DataLoader or by the wall-clock budget carry no loss, "
            "so none of them can be the best checkpoint."
        )
    return step_to_restore


def restore_train_state(checkpoint_manager: ocp.CheckpointManager, template: TrainState, step: int | None = None) -> TrainState:
    """Restore the training state from a checkpoint.

    Args:
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager that was used to save the checkpoint.
        template (TrainState): a template of the training state to restore. This should have the same structure as the training state that was saved.
        step (int | None): the step to restore. Defaults to None, which restores the best step.

    Returns:
        TrainState: the restored training state.
    """
    step_to_restore = _step_to_restore(checkpoint_manager, step)

    model_saveable, model_non_saveable = partition_saveable(template.model)
    opt_state_saveable, opt_state_non_saveable = partition_saveable(template.opt_state)

    composite_restore = ocp.args.Composite(
        model=ocp.args.StandardRestore(model_saveable),
        opt_state=ocp.args.StandardRestore(opt_state_saveable),
        metadata=ocp.args.JsonRestore(),
    )

    restored = checkpoint_manager.restore(
        step_to_restore,
        args=composite_restore,
    )

    model = eqx.combine(restored["model"], model_non_saveable)
    opt_state = eqx.combine(restored["opt_state"], opt_state_non_saveable)
    metadata = restored["metadata"]

    train_state = TrainState(step=metadata["step"], epoch=metadata["epoch"], model=model, opt_state=opt_state)
    return train_state


def restore_model(checkpoint_manager: ocp.CheckpointManager, template: TrainableModel, step: int | None = None) -> TrainableModel:
    """Restore just the model from a checkpoint.

    Args:
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager that was used to save the checkpoint.
        template (TrainableModel): a template of the model to restore. This should have the same structure as the model that was saved.
        step (int | None): the step to restore. Defaults to None, which restores the best step.

    Returns:
        TrainableModel: the restored model.
    """
    step_to_restore = _step_to_restore(checkpoint_manager, step)

    model_saveable, model_non_saveable = partition_saveable(template)

    model_saveable_restored = checkpoint_manager.restore(
        step_to_restore,
        args=ocp.args.Composite(
            model=ocp.args.StandardRestore(model_saveable),
        ),
    )["model"]

    model_out = eqx.combine(model_saveable_restored, model_non_saveable)
    return model_out


def restore_model_from_path(path: PathLike, template: TrainableModel) -> TrainableModel:
    """Restore a model from a checkpoint directory, given a template of the model. Note:the template should have the same structure as the model that was saved (e.g. same NN architecture, widths, etc).

    TrainState checkpoint directories should include a "model" subdirectory containing the model checkpoint. While training runs typically save the full TrainState, sometimes it is useful to just restore the model. This function provides a convenient way to do that.

    Args:
        path (PathLike): path to the directory containing just the model checkpoint.
        template (TrainableModel): a template of the model to restore. This should have the same structure as the model that was saved.

    Returns:
        TrainableModel: the restored model.
    """
    checkpointer = ocp.StandardCheckpointer()
    model_saveable, model_non_saveable = partition_saveable(template)

    model_saveable_restored = checkpointer.restore(
        path,
        model_saveable,
    )

    model_out = eqx.combine(model_saveable_restored, model_non_saveable)
    return model_out


def restore_train_state_from_path(path: PathLike, template: TrainState) -> TrainState:
    """Restore a TrainState from a checkpoint directory, given a template of the TrainState.

    Args:
        path (PathLike): path to the directory containing the TrainState checkpoint (e.g. the path you gave the CheckpointManager).
        template (TrainState): a template of the TrainState to restore. This should have the same structure as the TrainState that was saved.

    Returns:
        TrainState: the restored TrainState.
    """
    manager = create_default_checkpoint_manager(path)
    return restore_train_state(manager, template)
