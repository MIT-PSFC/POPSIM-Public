import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from popsim.ml.debug_utils import (
    check_large_model_weights,
    check_large_values,
    check_zero_variance_variables,
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
