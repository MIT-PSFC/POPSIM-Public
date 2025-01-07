import pytest
import jax
from popsim.jax_utils import jax_cpu, jax_gpu

def test_jax_cpu():
    def my_function(x):
        return x.sum()

    x = jax.numpy.arange(10)

    with jax_cpu():
        jitted_fn = jax.jit(my_function)(x)
        assert jitted_fn.device == jax.devices("cpu")[0]

def test_jax_gpu():
    def my_function(x):
        return x.sum()

    # Check if GPU backend is available
    if "gpu" not in [d.platform for d in jax.devices()]:
        pytest.skip("No GPU backend available")

    x = jax.numpy.arange(10)

    with jax_gpu():
        jitted_fn = jax.jit(my_function)(x)
        assert jitted_fn.device == jax.devices("gpu")[0]