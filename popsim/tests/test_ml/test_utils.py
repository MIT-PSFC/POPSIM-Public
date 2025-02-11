import pytest
import jax.numpy as jnp
import numpy as np
from popsim.ml.utils import pad_time, pad_time_xr
import xarray as xr

@pytest.mark.parametrize("times, expected", [
    (jnp.array([0., 1., 2., 3., 4., jnp.nan,jnp.nan]), 
     jnp.array([0., 1., 2., 3., 4., 4. + jnp.finfo(jnp.float64).eps, 4. + 2. * jnp.finfo(jnp.float64).eps])),
    (jnp.array([0., 1., 2., 3., 4., 4., 4.]), 
     jnp.array([0., 1., 2., 3., 4., 4. + jnp.finfo(jnp.float64).eps, 4. + 2. * jnp.finfo(jnp.float64).eps])),
    (jnp.array([1., 2., 3., 4., 5.]), 
     jnp.array([1., 2., 3., 4., 5.])),
    (jnp.array([1., 1., 1., 1.]), 
     jnp.array([1., 1. + 2. * jnp.finfo(jnp.float64).eps, 1. + 4. * jnp.finfo(jnp.float64).eps, 1. + 6. * jnp.finfo(jnp.float64).eps])),
    (jnp.array([1.]),
     jnp.array([1.])),
    (jnp.array([1., 2., 3., 3., 3., 4., 4.]), 
     jnp.array([1., 2., 3., 3. + 3. * jnp.finfo(jnp.float64).eps, 3. + 6. * jnp.finfo(jnp.float64).eps, 4., 4. + 3. * jnp.finfo(jnp.float64).eps])),
     
])
def test_pad_time(times, expected):
    result = pad_time(times)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    assert (result[1:] > result[:-1]).all()

@pytest.mark.parametrize("times", [
    jnp.array([2.11699986, 2.12600017, 2.13499999,
       2.14500022, 2.15400004, 2.16400027, 2.1730001 , 2.18299985,
       2.19200015, 2.20099998, 2.2110002 , 2.2110002 , 2.2110002]),
    jnp.array([1e10, np.nan, 1e11, 1e12, 1e12]),
    jnp.array([1e-200, 1e-200, 1e-200, 2e-200, 3e-200]),
    jnp.ones(10),
    jnp.array([0, 1, 2, 3, 4, jnp.nan, jnp.nan]),
    # TODO(ZanderKeith): jnp.zeros(10),
    # For the case of jnp.zeros(10), the function works but I don't know how to write a test for it
    # The > operator always returns False even though the array is strictly increasing.
    # Also, you can't do subtraction on floats too close to zero (~1e-324), it will just return zero.
])
def test_pad_time_only_inequality(times):
    out = pad_time(times)
    assert (out[1:] > out[:-1]).all()

def test_pad_time_xr():
    example_time = xr.DataArray(
        data=np.array([[0, 1, 2, 3, 4, 4, 4], [2, 3, 4, 5, 6, np.nan, np.nan]]),
        dims=["sample", "time"],
        coords={"sample": [0, 1]}
    )

    padded_time = pad_time_xr(example_time, "time")

    # Check that the time array is strictly increasing.
    assert (padded_time.diff("time") > 0.0).all()

    # Check that the time array is padded correctly.
    assert (padded_time.isel(sample=0).data == pad_time(example_time.isel(sample=0).data)).all()
    assert (padded_time.isel(sample=1).data == pad_time(example_time.isel(sample=1).data)).all()

    # Test that changing the order of sample and time, we still get the same result.
    padded_time2 = pad_time_xr(example_time.transpose(), "time")
    assert padded_time2.equals(padded_time)
