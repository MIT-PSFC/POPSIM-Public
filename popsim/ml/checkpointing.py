"""
Utilities for saving and loading model and training checkpoints.

Two checkpoint formats share one API (CheckpointFormat):
"orbax" saves each array item as a tensorstore OCDBT database, built for large and sharded models, about 40 files per checkpoint.
"minimal" saves each checkpoint as one uncompressed .npz file, written, pruned and restored by POPSIM alone.
Both hold the same items, metadata and loss, and convert_checkpoints rewrites one format as the other.
"""

import json
import os
import typing
from os import PathLike
from pathlib import Path

import chex
import equinox as eqx
import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from jaxtyping import PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.partition import PartitionFn

CheckpointFormat = typing.Literal["orbax", "minimal"]
CHECKPOINT_FORMATS: tuple[CheckpointFormat, ...] = typing.get_args(CheckpointFormat)
# A minimal checkpoint is the file <step> plus this suffix, written as <step>.npz.tmp until complete
MINIMAL_CHECKPOINT_SUFFIX = ".npz"
_MINIMAL_TMP_SUFFIX = ".tmp"
# Members of a minimal checkpoint beside its array leaves, each the JSON of a 0-d string array
_MINIMAL_ITEMS_MEMBER = "items"
_MINIMAL_METADATA_MEMBER = "metadata"
_MINIMAL_METRICS_MEMBER = "metrics"

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


def partition_saveable(pytree: PyTree) -> tuple[PyTree, PyTree]:
    """Partition a pytree into saveable and non-saveable parts."""
    return eqx.partition(pytree, eqx.is_array_like)


def _saveable_items(train_state: TrainState) -> dict[str, PyTree]:
    """The array items of a train state's checkpoint, the saveable partition of each.

    "model" is the evaluated model (evaluated_model) and "opt_state" the optimizer state.
    A train state that averages its weights adds the live iterate's trainable partition as RAW_TRAINABLE_ITEM.
    They are the values a save writes and the templates a restore fills.
    """
    model = evaluated_model(train_state)
    model_save, _ = partition_saveable(model)
    opt_state_save, _ = partition_saveable(train_state.opt_state)
    items = {"model": model_save, "opt_state": opt_state_save}
    if train_state.averaged_trainable is not None:
        raw_trainable = _leaves_at_positions_of(train_state.model, train_state.averaged_trainable)
        raw_trainable_save, _ = partition_saveable(raw_trainable)
        items[RAW_TRAINABLE_ITEM] = raw_trainable_save
    return items


def _unknown_format_error(checkpoint_format: str) -> ValueError:
    return ValueError(f"Unknown checkpoint format {checkpoint_format!r}, expected one of {CHECKPOINT_FORMATS}")


def checkpoint_step_path(directory: PathLike, step: int, checkpoint_format: CheckpointFormat) -> Path:
    """Where checkpoint step lives under directory, a dir in the orbax format and a file in the minimal one."""
    if checkpoint_format == "orbax":
        return Path(directory) / str(step)
    if checkpoint_format == "minimal":
        return Path(directory) / f"{step}{MINIMAL_CHECKPOINT_SUFFIX}"
    raise _unknown_format_error(checkpoint_format)


def saved_checkpoint_steps(directory: PathLike, checkpoint_format: CheckpointFormat) -> list[int]:
    """The steps of the finished checkpoints under directory, ascending, read from their names alone.

    Both formats write a checkpoint under a temporary name and rename it into place once complete,
    so an in-progress save is never listed.
    A missing directory holds none.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    if checkpoint_format == "orbax":
        step_names = [path.name for path in directory.iterdir() if path.is_dir()]
    elif checkpoint_format == "minimal":
        checkpoint_paths = [path for path in directory.iterdir() if path.is_file() and path.name.endswith(MINIMAL_CHECKPOINT_SUFFIX)]
        step_names = [path.name.removesuffix(MINIMAL_CHECKPOINT_SUFFIX) for path in checkpoint_paths]
    else:
        raise _unknown_format_error(checkpoint_format)
    steps = [int(step_name) for step_name in step_names if step_name.isdigit()]
    return sorted(steps)


class OrbaxCheckpointManager(ocp.CheckpointManager):
    """orbax's CheckpointManager with the item methods popsim saves and restores through.

    Each array item is a tensorstore OCDBT database, built for large and sharded models, about 40 files per checkpoint.
    MinimalCheckpointManager holds the same items in one file.
    """

    def save_items(self, step: int, items: dict[str, PyTree], metadata: dict, loss: float | None):
        """Save the array items and the metadata as checkpoint step, ranked by loss, a no-op for a step at or below the latest."""
        items_save = {item_name: ocp.args.StandardSave(item) for item_name, item in items.items()}
        composite_save = ocp.args.Composite(metadata=ocp.args.JsonSave(metadata), **items_save)
        # A None loss must be saved as no metrics, a None metric breaks the best-checkpoint ranking.
        metrics = {"loss": loss} if loss is not None else None
        self.save(step, args=composite_save, metrics=metrics)
        self.wait_until_finished()

    def restore_items(self, step: int, templates: dict[str, PyTree]) -> dict[str, PyTree]:
        """The array items of checkpoint step, each filling the saveable partition its template gives."""
        items_restore = {item_name: ocp.args.StandardRestore(template) for item_name, template in templates.items()}
        composite_restore = ocp.args.Composite(**items_restore)
        restored = self.restore(step, args=composite_restore)
        return {item_name: restored[item_name] for item_name in templates}

    def restore_metadata(self, step: int) -> dict:
        """The metadata dict saved with checkpoint step."""
        composite_restore = ocp.args.Composite(metadata=ocp.args.JsonRestore())
        restored = self.restore(step, args=composite_restore)
        return restored["metadata"]

    def holds_item(self, step: int, item_name: str) -> bool:
        """Whether checkpoint step saved the array item item_name."""
        item_dir = Path(self.directory) / str(step) / item_name
        return item_dir.is_dir()


class MinimalCheckpointManager:
    """Checkpoints as one uncompressed .npz file each, <step>.npz, written, pruned and restored by popsim alone.

    A file holds every leaf of each array item as the member "<item>/<jax keystr of the leaf's path>",
    the item names as "items", and the metadata dict and the metrics as JSON in "metadata" and "metrics",
    the same items an orbax checkpoint holds.
    numpy alone opens it and lists every array by name.
    It keeps and ranks checkpoints exactly as create_default_checkpoint_manager's orbax policy does (_prune).
    Single process only.
    """

    def __init__(self, directory: PathLike, max_to_keep: int = 1):
        process_count = jax.process_count()
        if process_count > 1:
            raise ValueError(f"The minimal checkpoint format writes from one process, this run has {process_count}. Use the orbax format.")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        # The metrics of every checkpoint on disk, None for one saved without a loss
        self._metrics: dict[int, dict | None] = {}
        for step in saved_checkpoint_steps(self.directory, "minimal"):
            self._metrics[step] = self._read_json_member(step, _MINIMAL_METRICS_MEMBER)

    def _path(self, step: int) -> Path:
        return checkpoint_step_path(self.directory, step, "minimal")

    def _read_json_member(self, step: int, member_name: str) -> typing.Any:
        checkpoint_path = self._path(step)
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint_file:
            member_json = checkpoint_file[member_name].item()
        return json.loads(member_json)

    def all_steps(self) -> list[int]:
        return sorted(self._metrics)

    def latest_step(self) -> int | None:
        return max(self._metrics, default=None)

    def _steps_worst_first(self) -> list[int]:
        """The steps saved with a loss, highest loss first, sorted as orbax's BestN and best_step sort them.

        The sort is stable, so of two equal losses the later step ranks better.
        """
        steps_with_loss = [step for step in self.all_steps() if self._metrics[step] is not None]
        return sorted(steps_with_loss, key=lambda step: _validation_loss(self._metrics[step]), reverse=True)

    def best_step(self) -> int | None:
        """The step with the lowest loss, None when no checkpoint has one."""
        steps_worst_first = self._steps_worst_first()
        return steps_worst_first[-1] if steps_worst_first else None

    def metrics(self, step: int) -> dict | None:
        """The metrics saved with checkpoint step, {"loss": loss}, or None when it was saved without a loss."""
        return self._metrics[step]

    def save_items(self, step: int, items: dict[str, PyTree], metadata: dict, loss: float | None):
        """Save the array items and the metadata as checkpoint step, ranked by loss, a no-op for a step at or below the latest.

        The file is written under a temporary name, synced, then renamed into place, so a partial file is never visible.
        """
        latest_step = self.latest_step()
        if latest_step is not None and step <= latest_step:
            return
        metrics = {"loss": loss} if loss is not None else None
        items_json = json.dumps(list(items))
        metadata_json = json.dumps(metadata)
        metrics_json = json.dumps(metrics)
        members = {
            _MINIMAL_ITEMS_MEMBER: np.array(items_json),
            _MINIMAL_METADATA_MEMBER: np.array(metadata_json),
            _MINIMAL_METRICS_MEMBER: np.array(metrics_json),
        }
        for item_name, item in items.items():
            leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(item)
            for leaf_path, leaf in leaves_with_paths:
                member_name = f"{item_name}/{jax.tree_util.keystr(leaf_path)}"
                members[member_name] = np.asarray(leaf)
        # A save cut short leaves its temporary file behind
        for stray_path in self.directory.glob(f"*{MINIMAL_CHECKPOINT_SUFFIX}{_MINIMAL_TMP_SUFFIX}"):
            stray_path.unlink()
        checkpoint_path = self._path(step)
        tmp_path = checkpoint_path.with_name(checkpoint_path.name + _MINIMAL_TMP_SUFFIX)
        with open(tmp_path, "wb") as tmp_file:
            np.savez(tmp_file, **members)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, checkpoint_path)
        self._metrics[step] = metrics
        self._prune()

    def _prune(self):
        """Delete the checkpoints orbax's AnyPreservationPolicy of LatestN(1) and BestN(max_to_keep) drops.

        Nothing is deleted while at most max_to_keep checkpoints exist.
        Beyond that the latest is kept, and the max_to_keep with the lowest loss, a checkpoint without one never counting.
        """
        if len(self._metrics) <= self.max_to_keep:
            return
        steps_worst_first = self._steps_worst_first()
        steps_to_keep = set(steps_worst_first[-self.max_to_keep :])
        steps_to_keep.add(self.latest_step())
        for step in self.all_steps():
            if step not in steps_to_keep:
                checkpoint_path = self._path(step)
                checkpoint_path.unlink()
                del self._metrics[step]

    def _saved_item_names(self, step: int) -> list[str]:
        return self._read_json_member(step, _MINIMAL_ITEMS_MEMBER)

    def restore_items(self, step: int, templates: dict[str, PyTree]) -> dict[str, PyTree]:
        """The array items of checkpoint step, each filling the saveable partition its template gives.

        Strict: an item must hold exactly the template's leaves, each of the template's shape and dtype, or it raises.
        """
        checkpoint_path = self._path(step)
        saved_item_names = self._saved_item_names(step)
        missing_items = [item_name for item_name in templates if item_name not in saved_item_names]
        if missing_items:
            raise KeyError(f"Checkpoint {checkpoint_path} holds the items {saved_item_names}, not {missing_items}")
        restored = {}
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint_file:
            for item_name, template in templates.items():
                restored[item_name] = _restored_minimal_item(checkpoint_file, item_name, template, checkpoint_path)
        return restored

    def restore_metadata(self, step: int) -> dict:
        """The metadata dict saved with checkpoint step."""
        return self._read_json_member(step, _MINIMAL_METADATA_MEMBER)

    def holds_item(self, step: int, item_name: str) -> bool:
        """Whether checkpoint step saved the array item item_name."""
        return item_name in self._saved_item_names(step)


def _restored_minimal_item(checkpoint_file: np.lib.npyio.NpzFile, item_name: str, template: PyTree, checkpoint_path: Path) -> PyTree:
    """The item item_name of an open minimal checkpoint, in template's structure, raising on any leaf mismatch."""
    member_prefix = f"{item_name}/"
    saved_member_names = {name for name in checkpoint_file.files if name.startswith(member_prefix)}
    template_leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    template_member_names = [member_prefix + jax.tree_util.keystr(leaf_path) for leaf_path, _ in template_leaves_with_paths]
    missing_names = sorted(set(template_member_names) - saved_member_names)
    unexpected_names = sorted(saved_member_names - set(template_member_names))
    if missing_names or unexpected_names:
        raise ValueError(
            f"Checkpoint {checkpoint_path} item {item_name!r} does not match the template, "
            f"missing {missing_names}, not in the template {unexpected_names}"
        )
    restored_leaves = []
    for member_name, (_, template_leaf) in zip(template_member_names, template_leaves_with_paths, strict=True):
        saved_array = checkpoint_file[member_name]
        restored_leaf = _restored_leaf(saved_array, template_leaf, member_name, checkpoint_path)
        restored_leaves.append(restored_leaf)
    return jax.tree_util.tree_unflatten(treedef, restored_leaves)


def _restored_leaf(saved_array: np.ndarray, template_leaf: typing.Any, member_name: str, checkpoint_path: Path) -> typing.Any:
    """saved_array as the kind of leaf template_leaf is, raising unless their shapes and dtypes match.

    A jax.Array (or an abstract ShapeDtypeStruct) lands on the template's sharding,
    a NumPy array or scalar stays NumPy, and a Python scalar becomes one.
    """
    template_array = template_leaf if hasattr(template_leaf, "dtype") else np.asarray(template_leaf)
    if saved_array.shape != template_array.shape or saved_array.dtype != template_array.dtype:
        raise ValueError(
            f"Checkpoint {checkpoint_path} holds {member_name} as {saved_array.dtype}{list(saved_array.shape)}, "
            f"the template as {template_array.dtype}{list(template_array.shape)}"
        )
    if isinstance(template_leaf, (jax.Array, jax.ShapeDtypeStruct)):
        return jax.device_put(saved_array, template_leaf.sharding)
    if isinstance(template_leaf, np.ndarray):
        return saved_array
    if isinstance(template_leaf, np.generic):
        return saved_array[()]
    return type(template_leaf)(saved_array.item())


CheckpointManager = OrbaxCheckpointManager | MinimalCheckpointManager


def create_default_checkpoint_manager(
    directory: PathLike, max_to_keep: int = 1, checkpoint_format: CheckpointFormat = "orbax"
) -> CheckpointManager:
    """Create the default CheckpointManager.

    It keeps the max_to_keep best checkpoints by validation loss, plus the latest checkpoint for resuming.
    Nothing is deleted while at most max_to_keep checkpoints exist.
    A checkpoint saved without a loss never counts as best,
    and is deleted once it is no longer the latest and more than max_to_keep checkpoints exist.
    Deletion only happens on save(), so a restore-only manager never deletes checkpoints.

    Args:
        directory (PathLike): the directory to save the checkpoints in.
        max_to_keep (int): how many best-by-validation-loss checkpoints to keep.
        checkpoint_format (CheckpointFormat): "orbax" (an OCDBT database per item, for large or sharded models)
            or "minimal" (one .npz file per checkpoint). Defaults to "orbax".

    Returns:
        CheckpointManager
    """
    if checkpoint_format == "minimal":
        return MinimalCheckpointManager(directory, max_to_keep=max_to_keep)
    if checkpoint_format != "orbax":
        raise _unknown_format_error(checkpoint_format)

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

    manager = OrbaxCheckpointManager(
        directory=directory,
        options=options,
        item_names=("model", "opt_state", "metadata", RAW_TRAINABLE_ITEM),
    )
    return manager


def save_train_state(
    train_state: TrainState,
    checkpoint_manager: CheckpointManager,
    loss: float | None,
    early_stopping_state: dict | None = None,
):
    """Save the training state to a checkpoint, a no-op if this epoch was already saved.

    The "model" item is the evaluated model (evaluated_model), the one every restore_model reads.
    A run that averages its weights also saves the live iterate's trainable partition as RAW_TRAINABLE_ITEM.

    Args:
        train_state (TrainState): the training state to save.
        checkpoint_manager (CheckpointManager): the checkpoint manager to save to.
        loss (float | None): the validation loss of the model, None for an epoch without validation.
        early_stopping_state (dict | None): the trainer's EarlyStopping.state(), None for a run without early stopping.
    """
    metadata = {
        "step": train_state.step,
        "epoch": train_state.epoch,
        EARLY_STOPPING_METADATA_KEY: early_stopping_state,
    }
    items = _saveable_items(train_state)
    checkpoint_manager.save_items(train_state.epoch, items, metadata, loss)


def _step_to_restore(checkpoint_manager: CheckpointManager, step: int | None) -> int:
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


def _check_weight_averaging_matches(checkpoint_manager: CheckpointManager, step: int, averages_weights: bool):
    """Raise unless the checkpoint at step was saved by a run that averages its weights exactly when the template does.

    An averaging run cannot resume a checkpoint without the live iterate,
    and a run without averaging would continue training from the averaged weights with the live iterate's optimizer state.
    """
    holds_raw_trainable = checkpoint_manager.holds_item(step, RAW_TRAINABLE_ITEM)
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


def restore_train_state(checkpoint_manager: CheckpointManager, template: TrainState, step: int | None = None) -> TrainState:
    """Restore the training state from a checkpoint.

    A template that averages its weights restores the live iterate from RAW_TRAINABLE_ITEM and the average from the "model" item.
    It must match the checkpoint, averaging both or neither.

    Args:
        checkpoint_manager (CheckpointManager): the checkpoint manager that was used to save the checkpoint.
        template (TrainState): a template of the training state to restore. This should have the same structure as the training state that was saved.
        step (int | None): the step to restore. Defaults to None, which restores the best step.

    Returns:
        TrainState: the restored training state.
    """
    step_to_restore = _step_to_restore(checkpoint_manager, step)
    averages_weights = template.averaged_trainable is not None
    _check_weight_averaging_matches(checkpoint_manager, step_to_restore, averages_weights)

    item_templates = _saveable_items(template)
    restored = checkpoint_manager.restore_items(step_to_restore, item_templates)
    metadata = checkpoint_manager.restore_metadata(step_to_restore)

    # The restored saveable partitions over the template's non-saveable leaves
    model_template = evaluated_model(template)
    model_evaluated = eqx.combine(restored["model"], model_template)
    opt_state = eqx.combine(restored["opt_state"], template.opt_state)

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


def restore_early_stopping_state(checkpoint_manager: CheckpointManager, step: int) -> dict | None:
    """The early-stopping state saved with a checkpoint, None when its run had no early stopping.

    Raises when the checkpoint carries no such state,
    so a resumed run never silently restarts its patience count.
    """
    metadata = checkpoint_manager.restore_metadata(step)
    if EARLY_STOPPING_METADATA_KEY not in metadata:
        raise KeyError(
            f"Checkpoint step {step} in {checkpoint_manager.directory} has no {EARLY_STOPPING_METADATA_KEY!r} metadata, "
            "so its run cannot resume with its patience count. Delete the checkpoint directory and retrain."
        )
    return metadata[EARLY_STOPPING_METADATA_KEY]


def restore_model(checkpoint_manager: CheckpointManager, template: TrainableModel, step: int | None = None) -> TrainableModel:
    """Restore just the model from a checkpoint, the averaged model when the run averaged its weights.

    Args:
        checkpoint_manager (CheckpointManager): the checkpoint manager that was used to save the checkpoint.
        template (TrainableModel): a template of the model to restore. This should have the same structure as the model that was saved.
        step (int | None): the step to restore. Defaults to None, which restores the best step.

    Returns:
        TrainableModel: the restored model.
    """
    step_to_restore = _step_to_restore(checkpoint_manager, step)
    model_saveable, _ = partition_saveable(template)
    restored = checkpoint_manager.restore_items(step_to_restore, {"model": model_saveable})
    return eqx.combine(restored["model"], template)


def restore_model_from_path(path: PathLike, template: TrainableModel) -> TrainableModel:
    """Restore a model from one orbax item directory, given a template of the model.
    Note: the template should have the same structure as the model that was saved (e.g. same NN architecture, widths, etc).

    Orbax format only: the path is the "model" subdirectory of an orbax checkpoint step.
    A minimal checkpoint restores its model through restore_model with a minimal manager.
    TrainState checkpoint directories should include a "model" subdirectory containing the model checkpoint.
    While training runs typically save the full TrainState, sometimes it is useful to just restore the model.
    This function provides a convenient way to do that.

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


def restore_train_state_from_path(path: PathLike, template: TrainState, checkpoint_format: CheckpointFormat = "orbax") -> TrainState:
    """Restore a TrainState from a checkpoint directory, given a template of the TrainState.

    Args:
        path (PathLike): path to the directory containing the TrainState checkpoint (e.g. the path you gave the CheckpointManager).
        template (TrainState): a template of the TrainState to restore. This should have the same structure as the TrainState that was saved.
        checkpoint_format (CheckpointFormat): the format the checkpoints were saved in. Defaults to "orbax".

    Returns:
        TrainState: the restored TrainState.
    """
    manager = create_default_checkpoint_manager(path, checkpoint_format=checkpoint_format)
    return restore_train_state(manager, template)


def convert_checkpoints(
    source_dir: PathLike,
    source_format: CheckpointFormat,
    destination_dir: PathLike,
    destination_format: CheckpointFormat,
    template: TrainState,
):
    """Rewrite every checkpoint in source_dir in destination_format under destination_dir.

    Each step keeps its array items bit for bit, its metadata (step, epoch, early-stopping state) and its loss,
    so all_steps, best_step, metrics and every restore agree between the two directories.
    Only orbax's own bookkeeping (timestamps, sharding and array metadata) is not carried, orbax writes it new on every save.
    The template supplies the pytree structure as for any restore,
    and must average its weights exactly when the checkpoints do.

    Args:
        source_dir (PathLike): the directory of the checkpoints to convert.
        source_format (CheckpointFormat): the format they were saved in.
        destination_dir (PathLike): the directory to write the converted checkpoints in, without checkpoints of destination_format.
        destination_format (CheckpointFormat): the format to write.
        template (TrainState): a template of the TrainState the checkpoints were saved from.
    """
    if not Path(source_dir).is_dir():
        raise FileNotFoundError(f"No checkpoint dir to convert at {source_dir}")
    existing_steps = saved_checkpoint_steps(destination_dir, destination_format)
    if existing_steps:
        raise FileExistsError(f"{destination_dir} already holds {destination_format} checkpoints at steps {existing_steps}")
    source_manager = create_default_checkpoint_manager(source_dir, checkpoint_format=source_format)
    source_steps = source_manager.all_steps()
    # At most max_to_keep checkpoints, so the destination prunes none of them
    destination_manager = create_default_checkpoint_manager(
        destination_dir, max_to_keep=max(len(source_steps), 1), checkpoint_format=destination_format
    )
    averages_weights = template.averaged_trainable is not None
    item_templates = _saveable_items(template)
    for step in source_steps:
        _check_weight_averaging_matches(source_manager, step, averages_weights)
        items = source_manager.restore_items(step, item_templates)
        metadata = source_manager.restore_metadata(step)
        metrics = source_manager.metrics(step)
        loss = _validation_loss(metrics) if metrics is not None else None
        destination_manager.save_items(step, items, metadata, loss)
