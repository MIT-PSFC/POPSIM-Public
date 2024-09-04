import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, ArrayLike, PyTree

# A loss function is a function that takes in a prediction and a target and returns a scalar loss.
Loss = typing.Callable[[PyTree[Array], PyTree[Array]], float]

# An instantaneous loss function is a function takes in a prediction and a target for a single time slice and returns a scalar loss.
InstantaneousLoss = typing.Callable[[PyTree[ArrayLike], PyTree[ArrayLike]], float]


class IntegralLoss(eqx.Module):
    """
    A wrapper to compute the integral of an instantaneous loss function over a time range.

    This class uses the trapezoidal rule to integrate a given instantaneous loss
    function over a specified time range. It can handle various strategies for
    dealing with NaN values in the loss calculations.

    Attributes:
        instantaneous_loss (InstantaneousLoss): An instance of a class that computes
            the instantaneous loss between predictions and targets.
        nan_strategy (str): The strategy to use when encountering NaN values.
            Options are:
            - "raise": Raise an error if NaN values are encountered (default).
            - "zero": Replace NaN values with zero.
            - "forward_fill": Fill NaN values with the last known non-NaN value.

    Args:
        instantaneous_loss (InstantaneousLoss): The instantaneous loss function to integrate.
        nan_strategy (str, optional): The strategy for handling NaN values. Defaults to "raise".

    Raises:
        ValueError: If an unknown nan_strategy is provided.

    Example:
        >>> def mse_loss(pred, target):
        ...     return ((pred - target) ** 2).mean()
        >>> loss_fn = IntegralLoss(mse_loss, nan_strategy="zero")
        >>> predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> targets = jnp.array([[1.1, 2.1], [3.1, 4.1]])
        >>> time = jnp.array([0.0, 1.0])
        >>> integral_loss = loss_fn(predictions, targets, time)
    """

    instantaneous_loss: InstantaneousLoss
    nan_strategy: str = eqx.field(static=True)

    def __init__(self, instantaneous_loss: InstantaneousLoss, nan_strategy: str = "raise"):
        self.instantaneous_loss = instantaneous_loss
        self.nan_strategy = nan_strategy

    def __call__(self, predictions: PyTree[Array], targets: PyTree[Array], time: Array) -> float:
        return _integral_loss(predictions, targets, time, self.instantaneous_loss, self.nan_strategy)


Loss = typing.Union[InstantaneousLoss, IntegralLoss]


def _forward_fill_nans(ts: Array, ys: Array) -> Array:
    ts2, ys2 = diffrax.rectilinear_interpolation(ts, ys)

    # For whatever reason, left=False still leaves the last value as a nan or inf.
    # To fix this, we add an extra point at the end with the maximum time and the last value.
    max_time = jnp.finfo(ts2.dtype).max

    ts2 = jnp.concatenate((ts2, jnp.array([max_time])))
    ys2 = jnp.concatenate((ys2, jnp.array([ys2[-1]])))

    interpolation = diffrax.LinearInterpolation(ts2, ys2)

    out = interpolation.evaluate(ts, left=False)

    return out


def _integral_loss(
    predictions: PyTree[Array],
    targets: PyTree[Array],
    time: Array,
    instantaneous_loss: InstantaneousLoss,
    nan_strategy: str,
) -> float:
    pred_spec = jax.tree.map(lambda _: 0, predictions)
    targ_spec = jax.tree.map(lambda _: 0, targets)
    instantaneous_values = jax.vmap(instantaneous_loss, in_axes=(pred_spec, targ_spec))(predictions, targets)

    if nan_strategy == "raise":
        eqx.error_if(instantaneous_values, jnp.isnan(instantaneous_values), "NaN values found in instantaneous loss values.")
    elif nan_strategy == "zero":
        filled = jnp.nan_to_num(instantaneous_values, nan=0.0)
        return trapezoid(filled, x=time)
    elif nan_strategy == "forward_fill":
        filled = _forward_fill_nans(time, instantaneous_values)
        return trapezoid(filled, x=time)
    else:
        raise ValueError(f"Unknown nan_strategy: {nan_strategy}")

    return trapezoid(instantaneous_values, x=time)
