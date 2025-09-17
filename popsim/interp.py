import typing
from enum import IntEnum

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from popsim.tree_util import tree_transpose


class InterpType(IntEnum):
    LINEAR = 0
    CUBIC = 1
    RECTILINEAR = 2


def interp(
    times: Array, tree: PyTree[Array], interp_type: InterpType = InterpType.LINEAR
) -> diffrax.LinearInterpolation | diffrax.CubicInterpolation:
    """Thin wrapper around diffrax.LinearInterpolation and diffrax.CubicInterpolation.

    Args:
        times (Array): times of the data.
        tree (PyTree[Array]): tree of arrays where the first dimension is the same as times.
        interp_type (InterpType, optional): interpolation interp_type.

    Raises:
        ValueError: invalid interp_type.

    Returns:
        typing.Union[diffrax.LinearInterpolation, diffrax.CubicInterpolation]: interpolation object.
    """

    if interp_type == InterpType.LINEAR:
        return diffrax.LinearInterpolation(ts=times, ys=tree)
    elif interp_type == InterpType.CUBIC:
        return diffrax.CubicInterpolation(ts=times, coeffs=diffrax.backward_hermite_coefficients(times, tree))
    elif interp_type == InterpType.RECTILINEAR:
        ts2, ys2 = diffrax.rectilinear_interpolation(times, tree)
        return diffrax.LinearInterpolation(ts2, ys2)
    else:
        raise ValueError(f"Unknown interp_type: {interp_type}")


def interp_time_dic(
    dic_trees: dict[float, PyTree[ArrayLike]], interp_type: InterpType
) -> PyTree[diffrax.LinearInterpolation] | PyTree[diffrax.CubicInterpolation]:
    """Given a dictionary where keys are times and values are trees, interpolate the trees at the times.

    Args:
        dic_trees (dict[float, PyTree[ArrayLike]]): A dictionary where keys are times and values are trees.
        interp_type (InterpType, optional): interpolation interp_type.

    Raises:
        ValueError: invalid interp_type.

    Returns:
        typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]: _description_
    """
    dic_trees = dict(sorted(dic_trees.items()))  # sort the dictionary by time
    times = jnp.array(list(dic_trees.keys()))
    trees = list(dic_trees.values())
    return interp_tree_seq(times, trees, interp_type)


def interp_tree_seq(
    times: Array, trees: typing.Sequence[PyTree[ArrayLike]], interp_type: InterpType
) -> PyTree[diffrax.LinearInterpolation] | PyTree[diffrax.CubicInterpolation]:
    """Given a sequence of times and a list of trees, interpolate the trees at the times.

    Args:
        times (Array): A sequence of times.
        trees (typing.Sequence[dict[float, PyTree[ArrayLike]]]): A list of trees to interpolate.
        interp_type (InterpType, optional): interpolation interp_type.

    Raises:
        ValueError: invalid interp_type.

    Returns:
        typing.Union[PyTree[diffrax.LinearInterpolation], PyTree[diffrax.CubicInterpolation]]: _description_
    """

    def interp_f(arr):
        return interp(times, arr, interp_type)

    trees_transposed = tree_transpose(trees)
    return jax.tree.map(interp_f, trees_transposed)


def resolve_paths(tree: PyTree[ArrayLike | diffrax.AbstractPath], t0: float, *args) -> PyTree[ArrayLike]:
    """Resolve any AbstractPath objects in a PyTree to their values at time t0.

    Args:
        tree (PyTree[Union[ArrayLike, AbstractPath]]): A PyTree of ArrayLike and AbstractPath objects.
        t0 (float): The time at which to evaluate the paths.

    Returns:
        PyTree[ArrayLike]: A PyTree of ArrayLike with all AbstractPath objects resolved to their values at time t0.
    """
    tree_resolved = jax.tree.map(
        lambda leaf: leaf.evaluate(t0, *args) if isinstance(leaf, diffrax.AbstractPath) else leaf,
        tree,
        is_leaf=lambda x: isinstance(x, diffrax.AbstractPath),
    )
    return tree_resolved


def interp_over_nans(ts: Array, data: PyTree[Array]) -> PyTree[Array]:
    """Perform a rectilinear interpolation over NaN values in the data.

    Args:
        ts (Array): time base.
        data (PyTree[Array]): data to interpolate.

    Returns:
        PyTree[Array]: interpolated data.
    """
    interp_fn = interp(ts, data, interp_type=InterpType.RECTILINEAR)
    result = interp_fn.evaluate(ts, left=True)
    return result
