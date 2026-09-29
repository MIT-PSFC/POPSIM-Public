import dataclasses
import os

import chex
import equinox as eqx
import jax
import numpy as np
import optax
import pytest

from popsim.ml.checkpointing import (
    TrainState,
    create_default_checkpoint_manager,
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

def test_checkpointing(tmpdir):
    nested_module = make_nested_module(10, key=0)
    train_state = TrainState.create_new(
        model=nested_module,
        partition_fn=lambda m: eqx.partition(m, eqx.is_inexact_array_like),
        optimizer=optax.adam(1e-3),
    )

    ckpt_dir = tmpdir / "ckpt"

    manager = create_default_checkpoint_manager(ckpt_dir)
    
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
        partition_fn=lambda m: eqx.partition(m, eqx.is_inexact_array_like),
        optimizer=optax.adam(1e-3),
    )
    train_state_lower_loss = dataclasses.replace(train_state_lower_loss, epoch=train_state.epoch + 1000)

    save_train_state(train_state_lower_loss, manager, loss=0.01)

    restored_train_state = restore_train_state(manager, train_state)

    chex.assert_trees_all_equal(train_state_lower_loss, restored_train_state)

    # Test thaat we can load the train_state from a path without using the manager directly.
    restored_train_state = restore_train_state_from_path(ckpt_dir, train_state)
    chex.assert_trees_all_equal(restored_train_state, train_state_lower_loss)


    # Test that we can load the model from a path without using the manager.
    model_path = os.path.join(ckpt_dir, str(train_state_lower_loss.epoch), "model")
    restored_model = restore_model_from_path(model_path, nested_module)
    chex.assert_trees_all_equal(restored_model, restored_train_state.model)


def _train_state_at_epoch(base: TrainState, epoch: int) -> TrainState:
    """A copy of base whose epoch and model weights both encode epoch, so a restore can be traced back."""
    model = {"w": np.full(3, float(epoch))}
    return dataclasses.replace(base, epoch=epoch, model=model)


def test_checkpoint_retention(tmpdir):
    """The default manager keeps the best max_to_keep checkpoints by loss plus the latest.

    Nothing is pruned while at most max_to_keep checkpoints exist.
    A checkpoint without a loss never counts as best,
    and is pruned once it is no longer the latest and more than max_to_keep checkpoints exist.
    Saving an epoch that is already on disk is a no-op.
    """
    base = TrainState.create_new(
        model={"w": np.zeros(3)},
        partition_fn=lambda m: eqx.partition(m, eqx.is_inexact_array_like),
        optimizer=optax.adam(1e-3),
    )
    ckpt_dir = tmpdir / "ckpt"
    manager = create_default_checkpoint_manager(ckpt_dir, max_to_keep=2)

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

    # A restore-only manager with a smaller max_to_keep never deletes anything.
    manager_reopened = create_default_checkpoint_manager(ckpt_dir, max_to_keep=1)
    assert manager_reopened.all_steps() == [3, 6]

    restored_best = restore_train_state(manager_reopened, base)
    chex.assert_trees_all_equal(restored_best, _train_state_at_epoch(base, 6))
    restored_step_3 = restore_train_state(manager_reopened, base, step=3)
    chex.assert_trees_all_equal(restored_step_3, _train_state_at_epoch(base, 3))
