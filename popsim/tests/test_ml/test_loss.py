import jax.numpy as jnp

from popsim.ml.loss import IntegralLoss
import pytest




# Hard-coded test cases
test_cases = [
    # Case 1: Without NaNs
    {
        "predictions": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "raise",
        "expected_result": 0.02,
        "expected_exception": None
    },
    # Case 2: With NaNs, using "zero" strategy
    {
        "predictions": jnp.array([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "zero",
        "expected_result": 0.01,
        "expected_exception": None
    },
    # Case 3: With NaNs, using "raise" strategy
    {
        "predictions": jnp.array([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "raise",
        "expected_result": None,
        "expected_exception": True
    },
    # Case 4: With NaNs, using "forward_fill" strategy
    {
        "predictions": jnp.array([[1.0, 2.0], [jnp.nan, 4.0], [5.0, 6.0]]),
        "targets": jnp.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]]),
        "time": jnp.array([0.0, 0.5, 1.0]),
        "nan_strategy": "forward_fill",
        "expected_result": 0.02,
        "expected_exception": None
    }
]

@pytest.mark.parametrize("case", test_cases)
def test_integral_loss(case):
    def l2loss(pred, target):
        return jnp.sum((pred - target) ** 2)
    integral_loss = IntegralLoss(l2loss, case["nan_strategy"])

    if case["expected_exception"]:
        # TODO(allenw): when jitted, this exception is not raised
        with pytest.raises(Exception):
            integral_loss(case["predictions"], case["targets"], case["time"])
    else:
        result = integral_loss(case["predictions"], case["targets"], case["time"])
        assert jnp.isclose(result, case["expected_result"], atol=1e-6)