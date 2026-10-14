"""
Utilities for saving and loading model and training checkpoints.
"""

import os
from os import PathLike
from pathlib import Path

import chex
import equinox as eqx
import jax
import optax
import orbax.checkpoint as ocp
from jaxtyping import PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.partition import PartitionFn

# Checkpoint metadata key of the early-stopping state, None for a run without early stopping
EARLY_STOPPING_METADATA_KEY = "early_stopping"
# Checkpoint item of the live iterate's trainable partition, saved only by a run that averages its weights.
# Its "model" item is then the averaged model, and a resumed run continues training from this item.
RAW_TRAINABLE_ITEM = "raw_trainable"


@chex.dataclass
class TrainState:
    step: int
    epoch: int
    model: TrainableModel
    opt_state: optax.OptState
    # The weight average of the trainable partition, None positions where the model is frozen.
    # None when the run does not average its weights.
    averaged_trainable: PyTree | None = None

    @classmethod
    def create_new(
        cls,
        model: TrainableModel,
        partition_fn: PartitionFn,
        optimizer: optax.GradientTransformation,
        average_weights: bool = False,
    ) -> "TrainState":
        trainable, _ = partition_fn(model)
        opt_state = optimizer.init(trainable)
        # The average starts at the init, which counts as its first iterate
        averaged_trainable = trainable if average_weights else None
        return cls(step=0, epoch=0, model=model, opt_state=opt_state, averaged_trainable=averaged_trainable)


def evaluated_model(train_state: TrainState) -> TrainableModel:
    """The model a train state is validated, checkpointed, restored and tested as.

    With weight averaging it is the averaged trainable leaves over the live model's frozen ones,
    so frozen buffers (normalizer statistics, frozen layers) are always the live model's.
    Without it, the live model itself.
    """
    if train_state.averaged_trainable is None:
        return train_state.model
    return eqx.combine(train_state.averaged_trainable, train_state.model)


def _leaves_at_positions_of(full_tree: PyTree, positions_template: PyTree) -> PyTree:
    """full_tree's leaves where positions_template holds a leaf, None where it holds None."""
    return jax.tree.map(
        lambda template_leaf, leaf: None if template_leaf is None else leaf,
        positions_template,
        full_tree,
        is_leaf=lambda leaf: leaf is None,
    )


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
        item_names=("model", "opt_state", "metadata", RAW_TRAINABLE_ITEM),
    )
    return manager


def partition_saveable(pytree: PyTree) -> tuple[PyTree, PyTree]:
    """Partition a pytree into saveable and non-saveable parts."""
    return eqx.partition(pytree, eqx.is_array_like)


def save_train_state(
    train_state: TrainState,
    checkpoint_manager: ocp.CheckpointManager,
    loss: float | None,
    early_stopping_state: dict | None = None,
):
    """Save the training state to a checkpoint, a no-op if this epoch was already saved.

    The "model" item is the evaluated model (evaluated_model), the one every restore_model reads.
    A run that averages its weights also saves the live iterate's trainable partition as RAW_TRAINABLE_ITEM.

    Args:
        train_state (TrainState): the training state to save.
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager to save to.
        loss (float | None): the validation loss of the model, None for an epoch without validation.
        early_stopping_state (dict | None): the trainer's EarlyStopping.state(), None for a run without early stopping.
    """
    metadata = {
        "step": train_state.step,
        "epoch": train_state.epoch,
        EARLY_STOPPING_METADATA_KEY: early_stopping_state,
    }

    model = evaluated_model(train_state)
    model_save, _ = partition_saveable(model)
    opt_state_save, _ = partition_saveable(train_state.opt_state)

    items_save = {
        "model": ocp.args.StandardSave(model_save),
        "opt_state": ocp.args.StandardSave(opt_state_save),
        "metadata": ocp.args.JsonSave(metadata),
    }
    if train_state.averaged_trainable is not None:
        raw_trainable = _leaves_at_positions_of(train_state.model, train_state.averaged_trainable)
        raw_trainable_save, _ = partition_saveable(raw_trainable)
        items_save[RAW_TRAINABLE_ITEM] = ocp.args.StandardSave(raw_trainable_save)
    composite_save = ocp.args.Composite(**items_save)

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


def _check_weight_averaging_matches(checkpoint_manager: ocp.CheckpointManager, step: int, averages_weights: bool):
    """Raise unless the checkpoint at step was saved by a run that averages its weights exactly when the template does.

    An averaging run cannot resume a checkpoint without the live iterate,
    and a run without averaging would continue training from the averaged weights with the live iterate's optimizer state.
    """
    step_dir = Path(checkpoint_manager.directory) / str(step)
    raw_trainable_dir = step_dir / RAW_TRAINABLE_ITEM
    holds_raw_trainable = raw_trainable_dir.is_dir()
    if averages_weights and not holds_raw_trainable:
        raise KeyError(
            f"Checkpoint step {step} in {checkpoint_manager.directory} has no {RAW_TRAINABLE_ITEM!r} item, "
            "so it was saved without weight averaging and cannot resume a run that averages. Delete the checkpoint directory and retrain."
        )
    if holds_raw_trainable and not averages_weights:
        raise ValueError(
            f"Checkpoint step {step} in {checkpoint_manager.directory} was saved with weight averaging, "
            "so its model is the average and only a run that averages can resume it. restore_model reads the averaged model."
        )


def restore_train_state(checkpoint_manager: ocp.CheckpointManager, template: TrainState, step: int | None = None) -> TrainState:
    """Restore the training state from a checkpoint.

    A template that averages its weights restores the live iterate from RAW_TRAINABLE_ITEM and the average from the "model" item.
    It must match the checkpoint, averaging both or neither.

    Args:
        checkpoint_manager (ocp.CheckpointManager): the checkpoint manager that was used to save the checkpoint.
        template (TrainState): a template of the training state to restore. This should have the same structure as the training state that was saved.
        step (int | None): the step to restore. Defaults to None, which restores the best step.

    Returns:
        TrainState: the restored training state.
    """
    step_to_restore = _step_to_restore(checkpoint_manager, step)
    averages_weights = template.averaged_trainable is not None
    _check_weight_averaging_matches(checkpoint_manager, step_to_restore, averages_weights)

    model_template = evaluated_model(template)
    model_saveable, model_non_saveable = partition_saveable(model_template)
    opt_state_saveable, opt_state_non_saveable = partition_saveable(template.opt_state)

    items_restore = {
        "model": ocp.args.StandardRestore(model_saveable),
        "opt_state": ocp.args.StandardRestore(opt_state_saveable),
        "metadata": ocp.args.JsonRestore(),
    }
    if averages_weights:
        raw_trainable_template = _leaves_at_positions_of(template.model, template.averaged_trainable)
        raw_trainable_saveable, _ = partition_saveable(raw_trainable_template)
        items_restore[RAW_TRAINABLE_ITEM] = ocp.args.StandardRestore(raw_trainable_saveable)

    composite_restore = ocp.args.Composite(**items_restore)
    restored = checkpoint_manager.restore(
        step_to_restore,
        args=composite_restore,
    )

    model_evaluated = eqx.combine(restored["model"], model_non_saveable)
    opt_state = eqx.combine(restored["opt_state"], opt_state_non_saveable)
    metadata = restored["metadata"]

    if averages_weights:
        # The live iterate at its trainable positions, the frozen leaves are the same in both models
        model = eqx.combine(restored[RAW_TRAINABLE_ITEM], model_evaluated)
        averaged_trainable = _leaves_at_positions_of(model_evaluated, template.averaged_trainable)
    else:
        model = model_evaluated
        averaged_trainable = None

    train_state = TrainState(
        step=metadata["step"],
        epoch=metadata["epoch"],
        model=model,
        opt_state=opt_state,
        averaged_trainable=averaged_trainable,
    )
    return train_state


def restore_early_stopping_state(checkpoint_manager: ocp.CheckpointManager, step: int) -> dict | None:
    """The early-stopping state saved with a checkpoint, None when its run had no early stopping.

    Raises when the checkpoint carries no such state,
    so a resumed run never silently restarts its patience count.
    """
    metadata = checkpoint_manager.restore(step, args=ocp.args.Composite(metadata=ocp.args.JsonRestore()))["metadata"]
    if EARLY_STOPPING_METADATA_KEY not in metadata:
        raise KeyError(
            f"Checkpoint step {step} in {checkpoint_manager.directory} has no {EARLY_STOPPING_METADATA_KEY!r} metadata, "
            "so its run cannot resume with its patience count. Delete the checkpoint directory and retrain."
        )
    return metadata[EARLY_STOPPING_METADATA_KEY]


def restore_model(checkpoint_manager: ocp.CheckpointManager, template: TrainableModel, step: int | None = None) -> TrainableModel:
    """Restore just the model from a checkpoint, the averaged model when the run averaged its weights.

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
