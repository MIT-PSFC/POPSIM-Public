import collections
from collections.abc import Sequence

import jax
import jax.numpy as jnp
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
    /home/awang/repos/POPSIM/popsim/interfaces
        Args:
            seq_of_trees (Sequence[PyTree[ScalarLike]]):

        Returns:
            PyTree[ArrayLike]: PyTree of arrays where the ith element of each array
                corresponds to the ith element of the input sequence.
    """
    return jax.tree_map(lambda *xs: jnp.array(xs), *seq_of_trees)


def leaves_as_array(tree: PyTree[ArrayLike]) -> Array:
    """Get the leaves of a PyTree as a single array.
    Note: this currently does not preserve dictionary order!
    https://github.com/google/jax/issues/4085

    Args:
        tree (PyTree[ArrayLike]): A PyTree of ArrayLike.

    Returns:
        Array: The leaves of the PyTree as a single array.
    """
    return jnp.atleast_1d(jnp.array(jax.tree_util.tree_leaves(tree)))


def build_ordered_dict(keys: Array, vals: Array) -> collections.OrderedDict:
    """Create an ordered dictionary from two arrays. Note: the usage of ordered dictionaries is important because Jax can
    accidentally sort non-ordered dictionaries. https://github.com/google/jax/issues/4085

    Args:
        keys (Array): keys for the dictionary.
        vals (Array): values for the dictionary.

    Returns:
        collections.OrderedDict: An ordered dictionary with keys and values.
    """
    return collections.OrderedDict(zip(keys, vals))
