import typing

import diffrax
import jax.numpy as jnp
from jaxtyping import PyTree, Real

from popsim.tree_util import tree_transpose


def interp_time_dic(
    dic_trees: dict[float, PyTree[Real]], interp_type: str = "linear"
) -> typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]:
    """Given a dictionary where keys are times and values are trees, interpolate the trees at the times.

    Args:
        dic_trees (dict[float, PyTree[Real]]): A dictionary where keys are times and values are trees.
        interp_type (str, optional): interpolation interp_type. Can be "linear" or "cubic". Defaults to "linear".

    Raises:
        ValueError: if interp_type is not "linear" or "cubic".

    Returns:
        typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]: _description_
    """
    times = jnp.array(list(dic_trees.keys()))
    trees = tree_transpose(list(dic_trees.values()))
    if interp_type == "linear":
        return diffrax.LinearInterpolation(ts=times, ys=trees)
    elif interp_type == "cubic":
        return diffrax.CubicInterpolation(ts=times, coeffs=diffrax.backward_hermite_coefficients(times, trees))
    else:
        raise ValueError(f"Unknown interp_type: {interp_type}")


def cubic_interp(ts, tree) -> diffrax.CubicInterpolation:
    return diffrax.CubicInterpolation(ts=ts, coeffs=diffrax.backward_hermite_coefficients(ts, tree))
