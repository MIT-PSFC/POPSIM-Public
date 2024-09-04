import typing

import jax
from jax.scipy.integrate import trapezoid
from jaxtyping import Array, ArrayLike, PyTree

# An instantaneous loss functiion is a function takes in a prediction and a target for a single time slice and returns a scalar loss.
InstantaneousLoss = typing.Callable[[PyTree[ArrayLike], PyTree[ArrayLike]], float]

# A loss function is a function that takes in a prediction and a target and returns a scalar loss.
Loss = typing.Callable[[PyTree[Array], PyTree[Array]], float]


class IntegralLoss:
    def __init__(self, instantaneous_loss: InstantaneousLoss):
        self.instantaneous_loss = instantaneous_loss

    def __call__(self, predictions: PyTree[Array], targets: PyTree[Array], time: Array) -> float:
        return integral_loss(predictions, targets, time, self.instantaneous_loss)


def integral_loss(
    predictions: PyTree[Array],
    targets: PyTree[Array],
    time: Array,
    instantaneous_loss: InstantaneousLoss,
) -> float:
    """Integrate a loss function over time. Expect the input predictions and targets to be PyTrees of arrays where the 0th axis corresponds to time.

    Args:
        predictions (PyTree[Array]):
        targets (PyTree[Array]): _description_
        time (Array): _description_
        instantaneous_loss (InstantaneousLoss): _description_

    Returns:

        float: _description_
    """
    pred_spec = jax.tree.map(lambda _: 0, predictions)
    targ_spec = jax.tree.map(lambda _: 0, targets)
    instantaneous_values = jax.vmap(instantaneous_loss, in_axes=(pred_spec, targ_spec))(predictions, targets)
    integral = trapezoid(instantaneous_values, x=time)
    return integral
