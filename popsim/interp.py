import typing

import diffrax
import jax.numpy as jnp
from jaxtyping import PyTree, Real

from popsim.tree_util import tree_transpose


def interp_time_dic(
    dic_trees: dict[float, PyTree[Real]], kind: str = "linear"
) -> typing.Union[diffrax.LinearInterpolation, diffrax.CubicInterpolation]:
    times = jnp.array(list(dic_trees.keys()))
    trees = tree_transpose(list(dic_trees.values()))
    if kind == "linear":
        return diffrax.LinearInterpolation(ts=times, ys=trees)
    elif kind == "cubic":
        return diffrax.CubicInterpolation(ts=times, coeffs=diffrax.backward_hermite_coefficients(times, trees))
    else:
        raise ValueError(f"Unknown kind: {kind}")


def cubic_interp(ts, tree) -> diffrax.CubicInterpolation:
    return diffrax.CubicInterpolation(ts=ts, coeffs=diffrax.backward_hermite_coefficients(ts, tree))
