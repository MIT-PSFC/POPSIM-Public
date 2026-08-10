import typing

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, ArrayLike, PyTree
from xarray_jax import dims_change_on_unflatten

from popsim.interp import interp_over_nans

# An instantaneous loss function is a function takes in a prediction and a target for a single time slice and returns a scalar loss.
InstantaneousLoss = typing.Callable[[PyTree[ArrayLike], PyTree[ArrayLike]], float]


class IntegralLoss(eqx.Module):
    r"""
    A wrapper class to compute the integral of a given loss function over a time range using the trapezoidal rule.

    Given a loss function $l(\mathfb{y}, \hat{\mathfb{y}})$ that computes the instantaneous loss between predictions, $\mathfb{y}$, and targets, $\hat{\mathfb{y}}$, wrapping $l$ with this class results in the following loss function:

    $$
    L(\mathfb{y}, \hat{\mathfb{y}}, \mathbf{t}) = \int_\mathbf{t} l(\mathbf{t}(t), \hat{\mathbf{t}}(t)) dt
    $$

    where $\mathbf{t}$ is a vector of time points.

    nan_strategy determines how to handle NaNs in the computed instantaneous loss values.


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

    def __init__(self, instantaneous_loss: InstantaneousLoss, nan_strategy: str = "ignore"):
        self.instantaneous_loss = instantaneous_loss
        self.nan_strategy = nan_strategy

    def __call__(self, predictions: PyTree[Array], targets: PyTree[Array], time: Array) -> float:
        return _integral_loss(predictions, targets, time, self.instantaneous_loss, self.nan_strategy)


LossFunction = InstantaneousLoss | IntegralLoss


def _integral_loss(
    predictions: PyTree[Array],
    targets: PyTree[Array],
    time: Array,
    instantaneous_loss: InstantaneousLoss,
    nan_strategy: str,
) -> float:
    # vmap over the leading (time) axis on flat leaves: any xarray types in the
    # predictions/targets are rebuilt inside the vmap with their leading dimension
    # dropped, which keeps dims consistent with the per-step data slices.
    leaves_pred, treedef_pred = jax.tree.flatten(predictions)
    leaves_targ, treedef_targ = jax.tree.flatten(targets)

    def _loss_at_step(leaves_p, leaves_t):
        with dims_change_on_unflatten(lambda dims: dims[1:]):
            step_pred = jax.tree.unflatten(treedef_pred, leaves_p)
            step_targ = jax.tree.unflatten(treedef_targ, leaves_t)
        return instantaneous_loss(step_pred, step_targ)

    instantaneous_values = jax.vmap(_loss_at_step, in_axes=(0, 0))(leaves_pred, leaves_targ)

    if nan_strategy == "raise":
        instantaneous_values = eqx.error_if(
            instantaneous_values, jnp.isnan(instantaneous_values), "NaN values found in instantaneous loss values."
        )
    elif nan_strategy == "zero":
        filled = jnp.nan_to_num(instantaneous_values, nan=0.0)
        return trapezoid(filled, x=time)
    elif nan_strategy == "forward_fill":
        filled = interp_over_nans(time, instantaneous_values)
        filled = eqx.error_if(filled, jnp.any(jnp.isnan(filled)), "NaN values found in filled instantaneous loss values.")
        time = eqx.error_if(time, jnp.any(jnp.isnan(time)), "NaN values found in time.")
        return trapezoid(filled, x=time)
    elif nan_strategy == "ignore":
        pass
    else:
        raise ValueError(f"Unknown nan_strategy: {nan_strategy}")

    out = trapezoid(instantaneous_values, x=time)
    return out
