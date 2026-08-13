import chex
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import pytest
import xarray as xr

from popsim.ml.dataloading import DataLoader, XarrayPreppedDataset
from popsim.ml.eval import (
    batch_loss,
    batched_model_eval_and_loss,
    eval_model_on_data,
    make_val_loss_eval_fn,
    model_eval_and_loss,
    run_evals,
)


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
    from popsim.modules.profile_predictor.train_configs import DUMMY_CONFIG
    from popsim.modules.profile_predictor.training_run_builder import ProfilePredictorTrainRunBuilder


    _, train_dl, val_dl, _ = ProfilePredictorTrainRunBuilder.get_dataloaders(DUMMY_CONFIG['dataloader_config'])
    model = ProfilePredictorTrainRunBuilder.model_init(train_dl, DUMMY_CONFIG['model_init_config'])
    loss_fn = ProfilePredictorTrainRunBuilder.get_loss_fn(DUMMY_CONFIG['loss_config'])

    # Test that we can evaluate the model on the training data.
    eval_data = eval_model_on_data(
        model=model,
        dataloader=val_dl
    )

    assert eval_data.input_ds.sample.size > 100
    assert eval_data.input_ds.sample.size == eval_data.output_ds.sample.size

    # Test that we can make a validation loss evaluation function and run it.
    val_loss_eval_fn = make_val_loss_eval_fn(loss_fn)
    val_loss = val_loss_eval_fn(eval_data)

    # Test we can use the val_loss function in run_evals.
    evals = run_evals(
        model=model,
        dataloader=val_dl,
        evaluation_suite={"val_loss": val_loss_eval_fn}
    )
    
    chex.assert_trees_all_close(evals["val_loss"], val_loss)

    # Check that evaluating on the train_dl returns the same result despite shuffling.
    eval_data_train = eval_model_on_data(
        model=model,
        dataloader=train_dl
    )
    eval_data_train2 = eval_model_on_data(
        model=model,
        dataloader=train_dl
    )
    xr.testing.assert_allclose(eval_data_train.output_ds, eval_data_train2.output_ds)

@pytest.mark.parametrize("convert_xr_to_jnp", [True, False])
def test_eval_shuffling(convert_xr_to_jnp):
    """Test that the evaluation function doesn't return scrambled outputs when the dataloader is shuffled."""
    from popsim.ml._types import TrainingMetadata

    class Module(eqx.Module):
        def __call__(self, x):
            return jax.tree.map(lambda x: x + 1.0, x)
        
    ds = xr.Dataset({'a': xr.DataArray(10.0 * jnp.arange(100), dims=['sample']), 'b': xr.DataArray(10.0 * jnp.arange(100) + 1, dims=['sample'])})

    ds = ds.assign_coords(sample=jnp.arange(100))

    ds = XarrayPreppedDataset(
        ds=ds,
        training_metadata=TrainingMetadata(
            sample_coord='sample',
            sample_dim='sample',
            input_vars=['a'],
            target_vars=['b'],
            convert_xr_to_jnp=convert_xr_to_jnp
        )
    )

    dl = DataLoader(
        dataset=ds,
        batch_size=10,
        shuffle=True,
    )

    eval_data = eval_model_on_data(
        model=Module(),
        dataloader=dl
    )

    assert eval_data.output_ds['a'].equals(eval_data.input_ds['b'])

    eval_data2 = eval_model_on_data(
        model=Module(),
        dataloader=dl
    )

    assert eval_data2.output_ds['a'].equals(eval_data2.input_ds['b'])

    assert eval_data2.output_ds.equals(eval_data.output_ds)