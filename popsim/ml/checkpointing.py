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


def create_default_checkpoint_manager(directory: PathLike, max_to_keep: int = 1) -> ocp.CheckpointManager:
    """Create the default CheckpointManager which saves the best checkpoint(s) based on the validation loss.

    Checkpoints beyond max_to_keep are deleted only when save() is called, so
    restore-only managers opened on an existing directory never garbage-collect
    checkpoints saved with a larger max_to_keep. Metrics are persisted per step,
    so best_step() on a reopened directory still picks the true best.

    Args:
        directory (PathLike): the directory to save the checkpoints in.
        max_to_keep (int): how many best-by-validation-loss checkpoints to keep.

    Returns:
        ocp.CheckpointManager: the CheckpointManager.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=1,
        best_fn=lambda val_metrics: val_metrics["loss"],
        best_mode="min",
    )

    manager = ocp.CheckpointManager(
        directory=directory,
        options=options,
        item_names=("model", "opt_state", "metadata"),
    )
    return manager


def create_latest_checkpoint_manager(directory: PathLike) -> ocp.CheckpointManager:
    """Create a CheckpointManager which only keeps the most recent checkpoint.

    Used for resuming interrupted training runs. Unlike the default manager,
    which keeps the best checkpoint by validation loss, this one always keeps
    the highest-epoch checkpoint so training can continue from where it left
    off. With no best_fn, orbax's best_step() falls back to latest_step(), so
    restore_train_state works with this manager unchanged.

    Args:
        directory (PathLike): the directory to save the checkpoints in.

    Returns:
        ocp.CheckpointManager: the CheckpointManager.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        save_interval_steps=1,
    )

    manager = ocp.CheckpointManager(
        directory=directory,
        options=options,
        item_names=("model", "opt_state", "metadata"),
    )
    return manager


def latest_checkpoint_dir(checkpoint_dir: PathLike) -> str:
    """Return the sibling directory used for latest-checkpoint (resume) saves."""
    return f"{os.fspath(checkpoint_dir)}_latest"


def partition_saveable(pytree: PyTree) -> tuple[PyTree, PyTree]:
    """Partition a pytree into saveable and non-saveable parts."""
    return eqx.partition(pytree, eqx.is_array_like)


def save_train_state(train_state: TrainState, checkpoint_manager: ocp.CheckpointManager, loss: float):
    """Save the training state to a checkpoint

    Args:
        train_state (TrainState): the training state to save.
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager to save to.
        loss (float): the loss value of the model.
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

    checkpoint_manager.save(
        train_state.epoch,
        args=composite_save,
        metrics={"loss": loss},
    )
    checkpoint_manager.wait_until_finished()


def restore_train_state(checkpoint_manager: ocp.CheckpointManager, template: TrainState) -> TrainState:
    """Restore the training state from a checkpoint.

    Args:
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager that was used to save the checkpoint.
        template (TrainState): a template of the training state to restore. This should have the same structure as the training state that was saved.

    Returns:
        TrainState: the restored training state.
    """
    step_to_restore = checkpoint_manager.best_step()

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
    step_to_restore = step if step is not None else checkpoint_manager.best_step()

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
