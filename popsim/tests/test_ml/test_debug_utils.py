import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from popsim.ml.dataloading import make_time_indep_dataloader
from popsim.ml.debug_utils import (
    check_large_model_weights,
    check_large_values,
    check_zero_variance_variables,
    diagnose_nans,
)


def test_check_large_model_weights():
    class SimpleModel(eqx.Module):
        w1: jnp.ndarray
        b1: jnp.ndarray
        w2: jnp.ndarray
        b2: jnp.ndarray

        def __init__(self):
            self.w1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
            self.b1 = jnp.array([0.5, -0.5])
            self.w2 = jnp.array([[10000.0, 2.0], [3.0, 4.0]])  # Large weight to trigger warning
            self.b2 = jnp.array([0.1, -0.1])

        def __call__(self, x):
            x = jnp.dot(x, self.w1) + self.b1
            x = jax.nn.relu(x)
            x = jnp.dot(x, self.w2) + self.b2
            return x

    model = SimpleModel()
    error_paths = check_large_model_weights(model, threshold=1e3)

    assert "w2" in error_paths
    assert len(error_paths) == 1

def test_check_large_values():
    data = {
        "normal_array": jnp.array([1.0, 2.0, 3.0]),
        "large_array": jnp.array([1.0, 2000.0, 3.0]),  # Large value to trigger warning
        "scalar": jnp.array(5.0),
        "large_scalar": jnp.array(5000.0),  # Large scalar to trigger warning
    }

    error_paths = check_large_values(data, threshold=1e3)

    assert "large_array" in error_paths
    assert "large_scalar" in error_paths
    assert len(error_paths) == 2

def test_check_zero_variance_variables():
    data = {
        "normal_array": jnp.array([1.0, 2.0, 3.0]),
        "zero_var_array": jnp.array([5.0, 5.0, 5.0]),  # Zero variance to trigger warning
        "scalar": jnp.array(10.0), # Scalar should be ignored
    }

    error_paths = check_zero_variance_variables(data)

    assert "zero_var_array" in error_paths
    assert "scalar" not in error_paths
    assert len(error_paths) == 1


def test_diagnose_nans_on_xarray_batch():
    """A batch built with convert_xr_to_jnp=False holds xr.Datasets, which xarray_jax registers as pytrees.

    The checks once rebuilt those Datasets around per-leaf booleans, which xarray_jax rejects,
    so diagnose_nans raised a ValueError in place of the TrainingDivergedError the trainer raises after it.
    """
    ds = xr.Dataset(
        {
            "x": (("episode", "time"), np.array([[1.0, 2.0, 3.0, 4.0]])),
            "constant": (("episode", "time"), np.array([[5.0, 5.0, 5.0, 5.0]])),
            "power": (("episode", "time"), np.array([[1.0, 2.0, 3.0, 5000.0]])),
        },
        coords={"episode": [0], "time": np.arange(4, dtype=float)},
    )
    dl = make_time_indep_dataloader(
        ds,
        time_coord="time",
        episode_coord="episode",
        input_vars=["x", "constant"],
        target_vars=["power"],
        shuffle=False,
        convert_xr_to_jnp=False,
    )
    batch = next(iter(dl))
    inputs, targets = batch.get_inputs_and_targets()
    assert isinstance(inputs, xr.Dataset)

    zero_variance_paths = check_zero_variance_variables(inputs)
    large_value_paths = check_large_values(targets, threshold=1e3)

    assert len(zero_variance_paths) == 1
    assert "constant" in zero_variance_paths[0]
    assert len(large_value_paths) == 1
    assert "power" in large_value_paths[0]
    diagnose_nans({"w": jnp.array(0.5)}, batch)
