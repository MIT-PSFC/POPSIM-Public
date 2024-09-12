import collections
import typing
from collections.abc import Sequence
from enum import Enum, IntEnum

import jax
import jax.numpy as jnp
import jax.tree_util as tu
from jaxtyping import Array, ArrayLike, PyTree, ScalarLike

import popsim.types as ptypes

"""
Much of Jax is designed around the concept of mapping over "PyTrees".
Thus, it is useful to have a library of common functions to manipulate them.
See the Jax documentation for more information:
    https://jax.readthedocs.io/en/latest/pytrees.html
"""


def get_instances_from_tree_leaves(tree: PyTree[typing.Any], type_: type) -> list[typing.Any]:
    """Get instances of a specific type from the leaves of a PyTree.

    Args:
        tree (PyTree[typing.Any]): A PyTree.
        type_ (type): The type to search for.

    Returns:
        list[typing.Any]: A list of instances of the specified type.
    """

    def func(x):
        return isinstance(x, type_)

    return [x for x in jax.tree.leaves(tree, is_leaf=func) if func(x)]


def tree_transpose_and_squeeze(seq_of_trees: Sequence[PyTree[ScalarLike]]) -> PyTree[ArrayLike]:
    """Transpose a sequence of pytrees into a single PyTree of arrays and squeeze the arrays to remove extraneous dimensions.

    Args:
        seq_of_trees (Sequence[PyTree[ScalarLike]]): A sequence of PyTrees.

    Returns:
        PyTree[ArrayLike]: PyTree of arrays where the ith element of each array
            corresponds to the ith element of the input sequence.
    """
    tree_transposed = tree_transpose(seq_of_trees)
    return jax.tree.map(lambda x: jnp.squeeze(x), tree_transposed)


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
    return jax.tree.map(lambda *xs: jnp.array(xs), *seq_of_trees)


def leaves_as_array(tree: PyTree[ArrayLike]) -> Array:
    """Get the leaves of a PyTree as a single array. Note: this currently does not preserve dictionary order! https://github.com/google/jax/issues/4085

    Args:
        tree (PyTree[ArrayLike]): A PyTree of ArrayLike.

    Returns:
        Array: The leaves of the PyTree as a single array.
    """
    return jnp.atleast_1d(jnp.array(jax.tree_util.tree_leaves(tree)))


def build_ordered_dict(keys: Array, vals: Array) -> collections.OrderedDict:
    """Create an ordered dictionary from two arrays. Note: the usage of ordered dictionaries is important because Jax can accidentally sort non-ordered dictionaries. https://github.com/google/jax/issues/4085

    Args:
        keys (Array): keys for the dictionary.
        vals (Array): values for the dictionary.

    Returns:
        collections.OrderedDict: An ordered dictionary with keys and values.
    """
    return collections.OrderedDict(zip(keys, vals))


def get_key(key: typing.Union[tu.SequenceKey, tu.DictKey, tu.GetAttrKey]) -> typing.Union[int, typing.Hashable, str]:
    """The different key types in Jax have different accessors. This is a wrapper function to get the key value.

    Args:
        key (typing.Union[tu.SequenceKey, tu.DictKey, tu.GetAttrKey]): key to access.

    Returns:
        typing.Union[int, typing.Hashable, str]: key value.
    """
    if isinstance(key, tu.SequenceKey):
        return key.idx
    elif isinstance(key, tu.DictKey):
        return key.key
    elif isinstance(key, tu.GetAttrKey):
        return key.name
    else:
        raise ValueError(f"Key type {type(key)} not recognized.")


def no_nans(tree: PyTree[ArrayLike]) -> bool:
    """Check for any NaNs in a PyTree's leaves.

    Args:
        tree (PyTree[ArrayLike]): the tree to check.

    Returns:
        bool: whether any NaNs are present.
    """
    # A tree with leaves that are True if the leaf is not NaN.
    not_nan_leaf_tree = jax.tree.map(lambda x: jnp.all(jnp.logical_not(jnp.isnan(x))), tree)

    leaves = jax.tree.leaves(not_nan_leaf_tree)

    return jnp.all(jnp.array(leaves))


def any_nans(tree: PyTree[typing.Any]) -> bool:
    """Check for any NaNs in a PyTree's leaves.

    Args:
        tree (PyTree[typing.Any]): the tree to check.

    Returns:
        bool: whether any NaNs are present.
    """
    return jnp.logical_not(no_nans(tree))


def keypath_to_string(keypath: tuple[ptypes.PyTreeKey]) -> str:
    """Convert a keypath to a string that looks pretty.

    Args:
        keypath (tuple[ptypes.PyTreeKey]): A keypath generated by e.g. jax.tree_util.tree_leaves_with_path.

    Returns:
        str: A string representation of the keypath.
    """

    def string_func(x: ptypes.PyTreeKey) -> str:
        key = get_key(x)
        if isinstance(key, (Enum, IntEnum)):
            # For enums, return the name of the class of the enum and the name of the enum value.
            # For example, an instance of Impurity.Tungsten would return "Impurity.Tungsten".
            type_name = key.__class__.__name__
            return f"{type_name}.{key.name}"
        return str(key)

    strings = [string_func(x) for x in keypath]

    # For some reason, there is often a leading dot in the keypath.
    strings = [x.lstrip(".") for x in strings]

    # The gui also does show quantities between <>.
    strings = [x.replace("<", "").replace(">", "") for x in strings]

    # Strip away brackets.
    strings = [x.replace("[", "").replace("]", "") for x in strings]

    # Finally strip away unnecessary quotes.
    strings = [x.replace("'", "") for x in strings]

    return ".".join(strings)
