import equinox as eqx
import jax.numpy as jnp
import pytest
import xarray as xr

from popsim.ml.loss import IntegralLoss

# Hard-coded test cases
test_cases = [
    # Case 0: Without NaNs
    {
        "predictions": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "raise",
        "expected_result": 0.02,
        "expected_exception": None,
        "loss_fn": "jnp",
    },
    # Case 1: With NaNs, using "zero" strategy
    {
        "predictions": jnp.array([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "zero",
        "expected_result": 0.01,
        "expected_exception": None,
        "loss_fn": "jnp",
    },
    # Case 2: With NaNs, using "raise" strategy
    {
        "predictions": jnp.array([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "raise",
        "expected_result": None,
        "expected_exception": True,
        "loss_fn": "jnp",
    },
    # Case 3: With NaNs, using "forward_fill" strategy
    {
        "predictions": jnp.array([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "forward_fill",
        "expected_result": 0.02,
        "expected_exception": None,
        "loss_fn": "jnp",
    },
    # Case 4: Without NaNs, two features which are 0D time-evolving scalars
    # Each leaf carries only the time dim, so inside the vmap each feature is a 0D scalar
    {
        "predictions": {
            "a": xr.DataArray([1.0, 3.0, 5.0], coords={"time": [0.0, 0.5, 1.0]}, dims=["time"]),
            "b": xr.DataArray([2.0, 4.0, 6.0], coords={"time": [0.0, 0.5, 1.0]}, dims=["time"]),
        },
        "targets": {
            "a": xr.DataArray([1.1, 3.1, 5.1], coords={"time": [0.0, 0.5, 1.0]}, dims=["time"]),
            "b": xr.DataArray([2.1, 4.1, 6.1], coords={"time": [0.0, 0.5, 1.0]}, dims=["time"]),
        },
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "raise",
        "expected_result": 0.02,
        "expected_exception": None,
        "loss_fn": "xarray_0d",
    },
    # Case 5: Without NaNs, one feature which is a 1D time-evolving array
    # The leaf carries (time, rho), so inside the vmap the feature is a 1D profile over rho
    {
        "predictions": xr.DataArray([[1.0, 1.1, 1.2, 1.3], [2.0, 2.1, 2.2, 2.3], [3.0, 3.1, 3.2, 3.3]], coords={"time": [0.0, 0.5, 1.0], "rho": [0.0, 0.25, 0.5, 1.0]}, dims=["time", "rho"]),
        "targets": xr.DataArray([[1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4], [3.1, 3.2, 3.3, 3.4]], coords={"time": [0.0, 0.5, 1.0], "rho": [0.0, 0.25, 0.5, 1.0]}, dims=["time", "rho"]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "raise",
        "expected_result": 0.01,
        "expected_exception": None,
        "loss_fn": "xarray_1d",
    },
]
@pytest.mark.parametrize("jitted", [False, True], ids=["eager", "jitted"])
@pytest.mark.parametrize("case", test_cases)
def test_integral_loss(case, jitted):
    def l2loss(pred, target):
        return jnp.sum((pred - target) ** 2)
    
    def l2loss_xarray_0d(pred, target):
        # pred/target are dicts of 0D DataArrays at this point, one per feature.
        return sum(jnp.sum((pred[key].data - target[key].data) ** 2) for key in pred)

    def l2loss_xarray_1d(pred, target):
        # pred/target are 1D DataArrays over rho at this point, so integrate the error over rho.
        return jnp.trapezoid((pred.data - target.data) ** 2, x=pred.coords["rho"].data)

    if case["loss_fn"] == "jnp":
        integral_loss = IntegralLoss(l2loss, case["nan_strategy"])
    elif case["loss_fn"] == "xarray_0d":
        integral_loss = IntegralLoss(l2loss_xarray_0d, case["nan_strategy"])
    elif case["loss_fn"] == "xarray_1d":
        integral_loss = IntegralLoss(l2loss_xarray_1d, case["nan_strategy"])

    if jitted:
        integral_loss = eqx.filter_jit(integral_loss)

    if case["expected_exception"]:
        with pytest.raises(Exception):
            integral_loss(case["predictions"], case["targets"], case["time"])
    else:
        result = integral_loss(case["predictions"], case["targets"], case["time"])
        assert jnp.isclose(result, case["expected_result"], atol=1e-6)