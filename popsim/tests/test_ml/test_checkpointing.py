import numpy as np
import jax
import equinox as eqx
import dataclasses
import chex
from popsim.ml.checkpointing import TrainState, save_train_state, create_default_checkpoint_manager, restore_model, restore_train_state, restore_model_from_path, restore_train_state_from_path
import optax

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
    model_path = ckpt_dir / str(train_state_lower_loss.epoch) / "model"
    restored_model = restore_model_from_path(model_path, nested_module)
    chex.assert_trees_all_equal(restored_model, restored_train_state.model)