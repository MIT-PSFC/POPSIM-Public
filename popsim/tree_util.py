from collections.abc import Sequence
from typing import Union

import jax
import jax.numpy as jnp
from diffrax import AbstractPath
from jaxtyping import Array, ArrayLike, PyTree, ScalarLike

"""
Much of Jax is designed around the concept of mapping over "PyTrees".
Thus, it is useful to have a library of common functions to manipulate them.
See the Jax documentation for more information:
    https://jax.readthedocs.io/en/latest/pytrees.html
"""


def tree_transpose(seq_of_trees: Sequence[PyTree[ScalarLike]]) -> PyTree[ArrayLike]:
    """Transpose a sequence of pytrees into a single PyTree of arrays.
    Consider the following super simple example:
        seq_of_trees = [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}]
    This is a sequence (list) of two PyTrees (in this case just dictionaries).
    Applying this function will produce:
        tree_of_arrays = tree_transpose(seq_of_trees)
    where:
        tree_of_arrays = {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])}

    Args:
        seq_of_trees (Sequence[PyTree[ScalarLike]]):

    Returns:
        PyTree[ArrayLike]: PyTree of arrays where the ith element of each array
            corresponds to the ith element of the input sequence.
    """
    return jax.tree_map(lambda *xs: jnp.array(xs), *seq_of_trees)


def leaves_as_array(tree: PyTree[ArrayLike]) -> Array:
    """Get the leaves of a PyTree as a single array.

    Args:
        tree (PyTree[ArrayLike]): A PyTree of ArrayLike.

    Returns:
        Array: The leaves of the PyTree as a single array.
    """
    return jnp.atleast_1d(jnp.array(jax.tree_util.tree_leaves(tree)))


def resolve_paths(tree: PyTree[Union[ArrayLike, AbstractPath]], t0: float, *args) -> PyTree[ArrayLike]:
    """Resolve any AbstractPath objects in a PyTree to their values at time t0.

    Args:
        tree (PyTree[Union[ArrayLike, AbstractPath]]): A PyTree of ArrayLike and AbstractPath objects.
        t0 (float): The time at which to evaluate the paths.

    Returns:
        PyTree[ArrayLike]: A PyTree of ArrayLike with all AbstractPath objects resolved to their values at time t0.
    """
    tree_resolved = jax.tree_map(
        lambda leaf: leaf.evaluate(t0, *args) if isinstance(leaf, AbstractPath) else leaf,
        tree,
        is_leaf=lambda x: isinstance(x, AbstractPath),
    )
    return tree_resolved
