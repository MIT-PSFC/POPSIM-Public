import typing

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from popsim.tree_util import tree_transpose


def interp_time_dic(
    dic_trees: dict[float, PyTree[ArrayLike]], interp_type: str = "linear"
) -> typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]:
    """Given a dictionary where keys are times and values are trees, interpolate the trees at the times.

    Args:
        dic_trees (dict[float, PyTree[ArrayLike]]): A dictionary where keys are times and values are trees.
        interp_type (str, optional): interpolation interp_type. Can be "linear" or "cubic". Defaults to "linear".

    Raises:
        ValueError: if interp_type is not "linear" or "cubic".

    Returns:
        typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]: _description_
    """
    times = jnp.array(list(dic_trees.keys()))
    trees = list(dic_trees.values())
    return interp_trees(times, trees, interp_type)


def interp_trees(
    times: Array, trees: typing.Sequence[PyTree[ArrayLike]], interp_type: str = "linear"
) -> typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]:
    """Given a sequence of times and a list of trees, interpolate the trees at the times.

    Args:
        times (Array): A sequence of times.
        trees (typing.Sequence[dict[float, PyTree[ArrayLike]]]): A list of trees to interpolate.
        interp_type (str, optional): interpolation interp_type. Can be "linear" or "cubic". Defaults to "linear".

    Raises:
        ValueError: if interp_type is not "linear" or "cubic".

    Returns:
        typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]: _description_
    """

    def interp_f(arr: Array) -> typing.Union[diffrax.LinearInterpolation, diffrax.CubicInterpolation]:
        if interp_type == "linear":
            return diffrax.LinearInterpolation(ts=times, ys=arr)
        elif interp_type == "cubic":
            return diffrax.CubicInterpolation(ts=times, coeffs=diffrax.backward_hermite_coefficients(times, arr))
        else:
            raise ValueError(f"Unknown interp_type: {interp_type}")

    trees_transposed = tree_transpose(trees)
    return jax.tree_map(interp_f, trees_transposed)


def resolve_paths(tree: PyTree[typing.Union[ArrayLike, diffrax.AbstractPath]], t0: float, *args) -> PyTree[ArrayLike]:
    """Resolve any AbstractPath objects in a PyTree to their values at time t0.

    Args:
        tree (PyTree[Union[ArrayLike, AbstractPath]]): A PyTree of ArrayLike and AbstractPath objects.
        t0 (float): The time at which to evaluate the paths.

    Returns:
        PyTree[ArrayLike]: A PyTree of ArrayLike with all AbstractPath objects resolved to their values at time t0.
    """
    tree_resolved = jax.tree_map(
        lambda leaf: leaf.evaluate(t0, *args) if isinstance(leaf, diffrax.AbstractPath) else leaf,
        tree,
        is_leaf=lambda x: isinstance(x, diffrax.AbstractPath),
    )
    return tree_resolved
