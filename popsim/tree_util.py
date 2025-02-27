import collections
import typing
from enum import Enum, IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np
import orbax.checkpoint as ocp
import xarray as xr
from jaxtyping import Array, ArrayLike, PyTree
from xarray_jax import var_change_on_unflatten

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


def tree_transpose(
    tree: typing.Union[list[PyTree[ArrayLike | xr.Variable | xr.DataArray]], PyTree[Array | xr.Variable | xr.DataArray]],
    extra_dim_name: typing.Optional[str] = None,
) -> typing.Union[PyTree[Array | xr.Variable | xr.DataArray], list[PyTree[ArrayLike | xr.Variable | xr.DataArray]]]:
    """Transpose back and forth between a list of PyTrees and a single PyTree of arrays. This function handles both the forward and inverse cases:

        1. (Forward) A list of PyTrees -> PyTree of arrays
        2. (Inverse) PyTree of arrays -> list of PyTrees

    It will automatically determine the input type and perform the appropriate transposition. Note that in the Inverse case, it is required that all leaves have the same size in the leading dimension. If there are xarray types in the tree, the dimension to add/remove must be specified.

    !!! Example

        ```python
        # List of PyTrees to PyTree of arrays
        seq_of_trees = [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}]
        tree_of_arrays = tree_transpose(seq_of_trees)
        # Result: {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])}

        # PyTree of arrays to list of PyTrees
        tree_of_arrays = {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])}
        seq_of_trees = tree_transpose(tree_of_arrays)
        # Result: [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}]

        # List of xr.DataArray to xr.DataArray
        seq_of_xr = [xr.DataArray([0.0, 1.0], dims="x"), xr.DataArray([2.0, 3.0], dims="x")]
        xr_array = tree_transpose(seq_of_xr, extra_dim_name="new_dim")
        # Result: xr.DataArray([[0.0, 1.0], [2.0, 3.0]], dims=("new_dim", "x"))
        ```

    Args:
        tree (typing.Union[list[PyTree[ArrayLike  |  xr.Variable  |  xr.DataArray]], PyTree[Array  |  xr.Variable  |  xr.DataArray]]): PyTree that can contain ArrayLike, xr.Variable, and xr.DataArray.
        extra_dim_name (typing.Optional[str], optional): The name of the extra dimension that gets added/removed when transposing between a list of PyTrees and a PyTree of arrays. Necessary if there are xarray types in the PyTree. Defaults to None.

    Returns:
        typing.Union[PyTree[Array | xr.Variable | xr.DataArray], list[PyTree[ArrayLike | xr.Variable | xr.DataArray]]]: transposed tree.
    """

    # Check if there are any xr.Variable instances.
    xr_vars = get_instances_from_tree_leaves(tree, xr.Variable)
    if len(xr_vars) > 0 and extra_dim_name is None:
        raise ValueError("The extra dimension name must be specified when there are xarray types in the tree.")

    def var_change_fn(var: xr.Variable):
        """The purpose of this function is to add or remove the extra dimension from the variable as need be."""
        ndims = len(var._dims)
        datadims = var._data.ndim
        if ndims == datadims:
            return var
        elif ndims == datadims - 1:
            newdims = (extra_dim_name, *var._dims)
            var._dims = newdims
            return var
        elif ndims == datadims + 1:
            newdims = tuple(d for d in var._dims if d != extra_dim_name)
            var._dims = newdims
            return var
        else:
            raise ValueError(f"Variable {var.name} has {ndims} dims but data has {datadims} dims.")

    with var_change_on_unflatten(var_change_fn):
        return _tree_transpose(tree)


def _tree_transpose(
    tree: typing.Union[list[PyTree[ArrayLike]], PyTree[Array]],
) -> typing.Union[PyTree[Array], list[PyTree[ArrayLike]]]:
    # Check if the input is a sequence of PyTrees
    if isinstance(tree, list):
        if len(tree) == 0:
            return {}

        def fun(*xs):
            return jnp.array(xs).squeeze()

        return jax.tree.map(fun, *tree)

    # If not a sequence, assume it's a PyTree of arrays
    elif isinstance(tree, PyTree):
        # Expect all leaves to have the same size in the leading dimension
        leaves = jax.tree.leaves(tree)

        # If there are no leaves, return an empty list.
        if len(leaves) == 0:
            return []

        # If all leaves are scalars, return the original tree in a list.
        if all(leaf.ndim == 0 for leaf in leaves):
            return [tree]

        # Check that all leaves have the same size in the leading dimension
        leaf_leading_sizes = [x.shape[0] for x in leaves]

        if not all(size == leaf_leading_sizes[0] for size in leaf_leading_sizes):
            raise ValueError("All leaves must have the same size in the leading dimension.")

        # Get the nummber of trees we need to create.
        n_trees = leaf_leading_sizes[0]

        if n_trees == 0:
            return [tree]

        # Create a function to extract the i-th tree.
        def extract_ith_element(i, tree):
            return jax.tree.map(lambda x: x[i], tree)

        # Use a list comprehension to create a list of trees
        return [extract_ith_element(i, tree) for i in range(n_trees)]


def leaves_as_array(tree: PyTree[ArrayLike]) -> Array:
    """Get the leaves of a PyTree as a single array. Note: this currently does not preserve dictionary order! https://github.com/google/jax/issues/4085

    Args:
        tree (PyTree[ArrayLike]): A PyTree of ArrayLike.

    Returns:
        Array: The leaves of the PyTree as a single array.
    """
    return jnp.atleast_1d(jnp.array(jax.tree.leaves(tree)))


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
    tree = eqx.filter(tree, eqx.is_array_like)
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


def to_json_compatible(tree: PyTree) -> dict:
    """Convert a PyTree to a JSON-compatible dictionary.

    Args:
        tree (PyTree): the tree to convert.

    Returns:
        dict: the JSON-compatible dictionary.
    """
    tree_dict = ocp.tree.serialize_tree(tree, keep_empty_nodes=True)

    # Convert all ArrayLike leaves to lists.
    # Keep valid types as they are. Valid types are from the JSON schema:
    #   https://json-schema.org/understanding-json-schema/reference/type
    # All other types are not JSON compatible, so we convert them to strings via repr.
    def convert(x):
        if isinstance(x, ArrayLike):
            return np.asarray(x).tolist()
        elif isinstance(x, (str, int, float, bool, type(None))):
            return x
        elif isinstance(x, Enum):
            return x.value
        else:
            return repr(x)

    tree_dict = jax.tree.map(convert, tree_dict)
    return tree_dict
