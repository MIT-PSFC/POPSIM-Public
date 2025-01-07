import jax.flatten_util
from popsim.ml.eval import model_eval_and_loss, batched_model_eval_and_loss, batch_loss, make_val_loss_eval_fn, run_evals, eval_model_on_data

import pytest
import equinox as eqx
import jax
import jax.numpy as jnp
import chex

@pytest.fixture
def simple_model():
    class Model(eqx.Module):
        number_add: float
        def __init__(self):
            self.number_add = 100.0

        def __call__(self, x):
            return jax.tree.map(lambda x: x + self.number_add, x)
    return Model()

def loss_fn(x, y):
    x_ravel, _ = jax.flatten_util.ravel_pytree(x)
    y_ravel, _ = jax.flatten_util.ravel_pytree(y)
    l2_norm = jnp.sum(jnp.square(x_ravel - y_ravel))
    return l2_norm

def test_eval_and_loss(simple_model):
    simple_input = {'a': jax.numpy.array([1, 2, 3]), 'b': jax.numpy.array([4, 5, 6])}
    number_add = simple_model.number_add
    simple_target = {'a': simple_input['a'] + number_add, 'b': simple_input['b'] + number_add}

    # Should be zero loss.
    assert model_eval_and_loss(simple_model, loss_fn, simple_input, simple_target) == 0.0

    # Now change the target to be off by 1.
    simple_target['a'] = simple_target['a'] + 1.0
    expected_loss = 1.0 * simple_input['a'].size
    assert model_eval_and_loss(simple_model, loss_fn, simple_input, simple_target) == expected_loss

    # Now test with batched input.
    simple_input = {'a': jax.numpy.array([[1, 2, 3], [4, 5, 6]]), 'b': jax.numpy.array([[7, 8, 9], [10, 11, 12]])}
    simple_target = {'a': simple_input['a'] + number_add, 'b': simple_input['b'] + number_add}
    simple_target['a'] = simple_target['a'] + 1.0
    losses = batched_model_eval_and_loss(simple_model, loss_fn, simple_input, simple_target)
    expected_losses = jnp.array([expected_loss, expected_loss])
    assert jnp.allclose(losses, expected_losses)

    # Now test batch_loss, which requires a partition first.
    trainable, static = eqx.partition(simple_model, eqx.is_inexact_array_like)
    assert batch_loss(trainable, static, loss_fn, simple_input, simple_target) == expected_loss

def test_eval():
    from popsim.modules.profile_predictor.train import get_training_objs
    from popsim.modules.profile_predictor.train_configs import SPARC_CONFIG

    trainer, _, val_dl, _ = get_training_objs(SPARC_CONFIG)
    model = trainer.train_state.model

    # Test that we can evaluate the model on the training data.
    eval_data = eval_model_on_data(
        model=model,
        dataloader=val_dl
    )

    assert eval_data.input_ds.sample.size > 100
    assert eval_data.input_ds.sample.size == eval_data.output_ds.sample.size

    # Test that we can make a validation loss evaluation function and run it.
    loss_fn = trainer.loss_fn
    val_loss_eval_fn = make_val_loss_eval_fn(loss_fn)
    val_loss = val_loss_eval_fn(eval_data)

    # Test we can use the val_loss function in run_evals.
    evals = run_evals(
        model=model,
        dataloader=val_dl,
        evaluation_suite={"val_loss": val_loss_eval_fn}
    )
    
    chex.assert_trees_all_close(evals["val_loss"], val_loss)
