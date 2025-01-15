import pytest
import jax.numpy as jnp
import numpy as np
from popsim.ml.utils import _count_repeat_elements, _repeat_time_hack

@pytest.mark.parametrize("times, expected", [
    (jnp.array([0, 1, 2, 3, 4, 4, 4]), jnp.array([0, 0, 0, 0, 0, 1, 2])),
    (jnp.array([1, 2, 3, 4, 5]), jnp.array([0, 0, 0, 0, 0])),
    (jnp.array([1, 1, 1, 1]), jnp.array([0, 1, 2, 3])),
    (jnp.array([1]), jnp.array([0])),
    (jnp.array([1, 2, 3, 3, 3, 4, 4]), jnp.array([0, 0, 0, 1, 2, 2, 3])),
])
def test_count_repeat_elements(times, expected):
    result = _count_repeat_elements(times)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize("times, eps_mult, expected", [
    (jnp.array([0., 1., 2., 3., 4., 4., 4.]), 1., 
     jnp.array([0., 1., 2., 3., 4., 4. + jnp.finfo(jnp.float64).eps, 4. + 2. * jnp.finfo(jnp.float64).eps])),
    (jnp.array([1., 2., 3., 4., 5.]), 1., 
     jnp.array([1., 2., 3., 4., 5.])),
    (jnp.array([1., 1., 1., 1.]), 2., 
     jnp.array([1., 1. + 2. * jnp.finfo(jnp.float64).eps, 1. + 4. * jnp.finfo(jnp.float64).eps, 1. + 6. * jnp.finfo(jnp.float64).eps])),
    (jnp.array([1.]), 1., 
     jnp.array([1.])),
    (jnp.array([1., 2., 3., 3., 3., 4., 4.]), 3., 
     jnp.array([1., 2., 3., 3. + 3. * jnp.finfo(jnp.float64).eps, 3. + 6. * jnp.finfo(jnp.float64).eps, 4., 4. + 3. * jnp.finfo(jnp.float64).eps])),
])
def test_repeat_time_hack(times, eps_mult, expected):
    result = _repeat_time_hack(times, eps_mult)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    assert (result[1:] > result[:-1]).all()

@pytest.mark.parametrize("times", [
    jnp.array([2.11699986, 2.12600017, 2.13499999,
       2.14500022, 2.15400004, 2.16400027, 2.1730001 , 2.18299985,
       2.19200015, 2.20099998, 2.2110002 , 2.2110002 , 2.2110002])]
)
def test_repeat_time_hack_only_inequality(times):
    out = _repeat_time_hack(times)
    assert (out[1:] > out[:-1]).all()