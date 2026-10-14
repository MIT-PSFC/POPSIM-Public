import dataclasses
import os

import chex
import equinox as eqx
import jax
import numpy as np
import optax
import pytest

from popsim.ml.checkpointing import (
    CHECKPOINT_FORMATS,
    TrainState,
    checkpoint_step_path,
    convert_checkpoints,
    create_default_checkpoint_manager,
    evaluated_model,
    restore_early_stopping_state,
    restore_model,
    restore_model_from_path,
    restore_train_state,
    restore_train_state_from_path,
    save_train_state,
)


class SubModule(eqx.Module):
    hi: np.ndarray

class Module(eqx.Module):
    a: np.ndarray
    b: dict
    c: SubModule
    nn: eqx.Module



def make_nested_module(size_param: int, key: int = 0):
    module = Module(a=np.arange(size_param), b={"a": np.arange(size_param), "b": np.arange(size_param)}, c=SubModule(hi=np.arange(size_param)), nn=eqx.nn.MLP(in_size=5, out_size=5, width_size=size_param, depth=2,key=jax.random.PRNGKey(key)))

    return module


def _partition_inexact(model):
    return eqx.partition(model, eqx.is_inexact_array_like)


@pytest.mark.parametrize("checkpoint_format", CHECKPOINT_FORMATS)
def test_checkpointing(tmpdir, checkpoint_format):
    nested_module = make_nested_module(10, key=0)
    train_state = TrainState.create_new(
        model=nested_module,
        partition_fn=_partition_inexact,
        optimizer=optax.adam(1e-3),
    )

    ckpt_dir = tmpdir / "ckpt"

    manager = create_default_checkpoint_manager(ckpt_dir, checkpoint_format=checkpoint_format)

    save_train_state(train_state, manager, loss=1.0)


    # Test that we can restore the train_state. Begin by messing up the epoch number.
    train_state_new = dataclasses.replace(train_state, epoch=train_state.epoch + 100)
    restored_train_state = restore_train_state(manager, train_state_new)
    chex.assert_trees_all_equal(train_state, restored_train_state)


    # Test that we can just restore the model. Begin by initializing the NN with a different key.
    new_nested_module = make_nested_module(10, key=42)
    module_restored = restore_model(manager, new_nested_module)
    chex.assert_trees_all_equal(nested_module, module_restored)


    # Test that if we save a new train_state with a lower loss, then the restored train_state will be the model with a lower loss.
    train_state_lower_loss = TrainState.create_new(
        model=make_nested_module(10, key=-42),
        partition_fn=_partition_inexact,
        optimizer=optax.adam(1e-3),
    )
    train_state_lower_loss = dataclasses.replace(train_state_lower_loss, epoch=train_state.epoch + 1000)

    save_train_state(train_state_lower_loss, manager, loss=0.01)

    restored_train_state = restore_train_state(manager, train_state)

    chex.assert_trees_all_equal(train_state_lower_loss, restored_train_state)

    # Test thaat we can load the train_state from a path without using the manager directly.
    restored_train_state = restore_train_state_from_path(ckpt_dir, train_state, checkpoint_format=checkpoint_format)
    chex.assert_trees_all_equal(restored_train_state, train_state_lower_loss)


@pytest.mark.parametrize("average_weights", [False, True], ids=["raw", "averaged"])
def test_restore_model_from_path_reads_orbax_model_item(tmpdir, average_weights):
    """restore_model_from_path reads the "model" item dir of an orbax step, the averaged model when the run averages."""
    train_state = TrainState.create_new(make_nested_module(10, key=0), _partition_inexact, optax.adam(1e-3), average_weights=average_weights)
    if average_weights:
        averaged_trainable = jax.tree.map(lambda leaf: leaf + 1.0, train_state.averaged_trainable)
        train_state = dataclasses.replace(train_state, averaged_trainable=averaged_trainable)
    ckpt_dir = tmpdir / "ckpt"
    manager = create_default_checkpoint_manager(ckpt_dir)
    save_train_state(train_state, manager, loss=1.0)

    model_path = os.path.join(ckpt_dir, str(train_state.epoch), "model")
    model_restored = restore_model_from_path(model_path, make_nested_module(10, key=42))
    chex.assert_trees_all_equal(model_restored, evaluated_model(train_state))


def _train_state_at_epoch(base: TrainState, epoch: int) -> TrainState:
    """A copy of base whose epoch and model weights both encode epoch, so a restore can be traced back."""
    model = {"w": np.full(3, float(epoch))}
    return dataclasses.replace(base, epoch=epoch, model=model)


@pytest.mark.parametrize("checkpoint_format", CHECKPOINT_FORMATS)
def test_checkpoint_retention(tmpdir, checkpoint_format):
    """The default manager keeps the best max_to_keep checkpoints by loss plus the latest.

    Nothing is pruned while at most max_to_keep checkpoints exist.
    A checkpoint without a loss never counts as best,
    and is pruned once it is no longer the latest and more than max_to_keep checkpoints exist.
    Saving an epoch that is already on disk is a no-op.
    """
    base = TrainState.create_new(
        model={"w": np.zeros(3)},
        partition_fn=_partition_inexact,
        optimizer=optax.adam(1e-3),
    )
    ckpt_dir = tmpdir / "ckpt"
    manager = create_default_checkpoint_manager(ckpt_dir, max_to_keep=2, checkpoint_format=checkpoint_format)

    # epoch, loss, steps on disk after the save, best step after the save
    save_sequence = [
        (1, None, [1], None),
        (2, 3.0, [1, 2], 2),
        (3, 1.0, [2, 3], 3),
        (4, 2.0, [3, 4], 3),
        (5, None, [3, 4, 5], 3),
        (5, 0.5, [3, 4, 5], 3),
        (6, 0.1, [3, 6], 6),
    ]
    for epoch, loss, expected_steps, expected_best in save_sequence:
        train_state = _train_state_at_epoch(base, epoch)
        save_train_state(train_state, manager, loss=loss)
        assert manager.all_steps() == expected_steps, (epoch, loss)
        assert manager.best_step() == expected_best, (epoch, loss)
        if expected_best is None:
            # Orbax would silently fall back to the unvalidated latest step.
            with pytest.raises(FileNotFoundError):
                restore_train_state(manager, base)

    # Each checkpoint is one entry of the directory, a dir in the orbax format and a file in the minimal one
    expected_names = [checkpoint_step_path(ckpt_dir, step, checkpoint_format).name for step in (3, 6)]
    assert sorted(os.listdir(ckpt_dir)) == expected_names

    # A restore-only manager with a smaller max_to_keep never deletes anything.
    manager_reopened = create_default_checkpoint_manager(ckpt_dir, max_to_keep=1, checkpoint_format=checkpoint_format)
    assert manager_reopened.all_steps() == [3, 6]

    restored_best = restore_train_state(manager_reopened, base)
    chex.assert_trees_all_equal(restored_best, _train_state_at_epoch(base, 6))
    restored_step_3 = restore_train_state(manager_reopened, base, step=3)
    chex.assert_trees_all_equal(restored_step_3, _train_state_at_epoch(base, 3))


@pytest.mark.parametrize("checkpoint_format", CHECKPOINT_FORMATS)
def test_averaged_train_state_round_trip(tmpdir, checkpoint_format):
    """An averaging train state restores its live model and its average exactly.

    Its "model" item is the averaged model, which restore_model reads.
    A template must average exactly when the checkpoint does, either mismatch raises.
    """
    optimizer = optax.adam(1e-3)
    train_state = TrainState.create_new(make_nested_module(10, key=0), _partition_inexact, optimizer, average_weights=True)
    averaged_trainable = jax.tree.map(lambda leaf: leaf + 1.0, train_state.averaged_trainable)
    train_state = dataclasses.replace(train_state, step=7, epoch=3, averaged_trainable=averaged_trainable)
    ckpt_dir = tmpdir / "ckpt"
    manager = create_default_checkpoint_manager(ckpt_dir, checkpoint_format=checkpoint_format)
    save_train_state(train_state, manager, loss=1.0)

    template = TrainState.create_new(make_nested_module(10, key=42), _partition_inexact, optimizer, average_weights=True)
    restored_train_state = restore_train_state(manager, template)
    chex.assert_trees_all_equal(restored_train_state, train_state)

    model_evaluated = evaluated_model(train_state)
    model_restored = restore_model(manager, template.model)
    chex.assert_trees_all_equal(model_restored, model_evaluated)

    template_raw = TrainState.create_new(make_nested_module(10, key=42), _partition_inexact, optimizer)
    with pytest.raises(ValueError, match="saved with weight averaging"):
        restore_train_state(manager, template_raw)

    manager_raw = create_default_checkpoint_manager(tmpdir / "ckpt_raw", checkpoint_format=checkpoint_format)
    save_train_state(template_raw, manager_raw, loss=1.0)
    with pytest.raises(KeyError, match="raw_trainable"):
        restore_train_state(manager_raw, template)


def test_minimal_restore_rejects_mismatched_template(tmpdir):
    """A minimal restore raises unless the template holds exactly the saved leaves, each of the saved shape and dtype."""
    model = {"w": np.zeros(3), "b": np.zeros(2)}
    train_state = TrainState.create_new(model, _partition_inexact, optax.adam(1e-3))
    manager = create_default_checkpoint_manager(tmpdir / "ckpt", checkpoint_format="minimal")
    save_train_state(train_state, manager, loss=1.0)

    with pytest.raises(ValueError, match=r"\['b'\]"):
        restore_model(manager, {"w": np.zeros(3)})
    with pytest.raises(ValueError, match=r"\['c'\]"):
        restore_model(manager, {"w": np.zeros(3), "b": np.zeros(2), "c": np.zeros(1)})
    with pytest.raises(ValueError, match=r"\['w'\]"):
        restore_model(manager, {"w": np.zeros(4), "b": np.zeros(2)})
    with pytest.raises(ValueError, match=r"\['w'\]"):
        restore_model(manager, {"w": np.zeros(3, dtype=np.float32), "b": np.zeros(2)})


def test_convert_checkpoints_round_trip(tmpdir):
    """orbax to minimal and back keeps every step's train state, metrics and early-stopping state.

    The steps include one saved without a loss and a best step that is not the latest.
    """
    optimizer = optax.adam(1e-3)
    template = TrainState.create_new(make_nested_module(10, key=42), _partition_inexact, optimizer, average_weights=True)
    orbax_dir = tmpdir / "orbax"
    manager_orbax = create_default_checkpoint_manager(orbax_dir, max_to_keep=4)
    epoch_losses = {1: None, 2: 0.5, 3: 0.2, 4: 0.7}
    saved_train_states = {}
    saved_early_stopping_states = {}
    for epoch, loss in epoch_losses.items():
        train_state = TrainState.create_new(make_nested_module(10, key=epoch), _partition_inexact, optimizer, average_weights=True)
        averaged_trainable = jax.tree.map(lambda leaf, shift=epoch: leaf + shift, train_state.averaged_trainable)
        train_state = dataclasses.replace(train_state, step=10 * epoch, epoch=epoch, averaged_trainable=averaged_trainable)
        early_stopping_state = {"best_loss": 0.1 * epoch, "counter": epoch}
        save_train_state(train_state, manager_orbax, loss=loss, early_stopping_state=early_stopping_state)
        saved_train_states[epoch] = train_state
        saved_early_stopping_states[epoch] = early_stopping_state

    minimal_dir = tmpdir / "minimal"
    orbax_again_dir = tmpdir / "orbax_again"
    convert_checkpoints(orbax_dir, "orbax", minimal_dir, "minimal", template)
    convert_checkpoints(minimal_dir, "minimal", orbax_again_dir, "orbax", template)
    assert sorted(os.listdir(minimal_dir)) == ["1.npz", "2.npz", "3.npz", "4.npz"]

    for checkpoint_dir, checkpoint_format in [(minimal_dir, "minimal"), (orbax_again_dir, "orbax")]:
        manager = create_default_checkpoint_manager(checkpoint_dir, checkpoint_format=checkpoint_format)
        assert manager.all_steps() == [1, 2, 3, 4]
        assert manager.best_step() == 3
        for epoch, loss in epoch_losses.items():
            restored_train_state = restore_train_state(manager, template, step=epoch)
            chex.assert_trees_all_equal(restored_train_state, saved_train_states[epoch])
            expected_metrics = {"loss": loss} if loss is not None else None
            assert manager.metrics(epoch) == expected_metrics
            assert restore_early_stopping_state(manager, epoch) == saved_early_stopping_states[epoch]
