import pytest
import jax.numpy as jnp
import numpy as np
from popsim.ml.utils import count_repeat_elements, repeat_time_hack

@pytest.mark.parametrize("times, expected", [
    (jnp.array([0, 1, 2, 3, 4, 4, 4]), jnp.array([0, 0, 0, 0, 0, 1, 2])),
    (jnp.array([1, 2, 3, 4, 5]), jnp.array([0, 0, 0, 0, 0])),
    (jnp.array([1, 1, 1, 1]), jnp.array([0, 1, 2, 3])),
    (jnp.array([1]), jnp.array([0])),
    (jnp.array([1, 2, 3, 3, 3, 4, 4]), jnp.array([0, 0, 0, 1, 2, 0, 1])),
])
def test_count_repeat_elements(times, expected):
    result = count_repeat_elements(times)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("times, eps_mult, expected", [
    (jnp.array([0, 1, 2, 3, 4, 4, 4]), 1, 
     jnp.array([0, 1, 2, 3, 4, 4 + jnp.finfo(jnp.float32).eps, 4 + 2 * jnp.finfo(jnp.float32).eps])),
    (jnp.array([1, 2, 3, 4, 5]), 1, 
     jnp.array([1, 2, 3, 4, 5])),
    (jnp.array([1, 1, 1, 1]), 2, 
     jnp.array([1, 1 + 2 * jnp.finfo(jnp.float32).eps, 1 + 4 * jnp.finfo(jnp.float32).eps, 1 + 6 * jnp.finfo(jnp.float32).eps])),
    (jnp.array([1]), 1, 
     jnp.array([1])),
    (jnp.array([1, 2, 3, 3, 3, 4, 4]), 3, 
     jnp.array([1, 2, 3, 3 + 3 * jnp.finfo(jnp.float32).eps, 3 + 6 * jnp.finfo(jnp.float32).eps, 4, 4 + 3 * jnp.finfo(jnp.float32).eps])),
])
def test_repeat_time_hack(times, eps_mult, expected):
    result = repeat_time_hack(times, eps_mult)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)