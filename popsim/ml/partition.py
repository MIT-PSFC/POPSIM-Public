import typing
from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import PyTree

"""
Partition functions and helpers. A partition function is a function that takes in a PyTree and splits that PyTree into two disjoint PyTrees. That is:
    partition_fn(tree) -> (tree1, tree2)
where the union of tree1 and tree2 is equal to tree, and tree1 and tree2 are disjoint.

This operation is very helpful for situations where you want to train only a subset of the model's parameters, or when you want to apply different transformations to different parts of the model.
"""


# Define a function that partitions a PyTree into two PyTrees.
PartitionFn = typing.Callable[[PyTree], tuple[PyTree, PyTree]]


def partition_by_arraylike(pytree: PyTree) -> tuple[PyTree, PyTree]:
    """Partition a PyTree into two PyTrees based on whether the elements are array-like.

    Args:
        pytree (PyTree): The PyTree to partition.

    Returns:
        tuple[PyTree, PyTree]: A tuple of two PyTrees. The first PyTree contains the array-like elements of the input PyTree, and the second PyTree contains the non-array-like elements of the input PyTree.
    """

    return eqx.partition(pytree, eqx.is_array_like)


def make_partition_by_members(
    member_getter: typing.Callable[[PyTree], PyTree],
    leaf_filter: Callable[[typing.Any], bool] = eqx.is_inexact_array,
) -> PartitionFn:
    """Generate a partition function that partitions a PyTree into two PyTrees, where the first contains the user specified members and the second contains everything else. The primary use case of this function is to allow the user to specify subsets of a model to train (e.g. perhaps you only want to tune some of the coefficients in a power law, but not all of them).

    More precisely, the user specifies the members that belong to the first tree using the "member_getter" function and the "leaf_filter" function. The "member_getter" function takes a PyTree and returns a tuple of objects that are members of the first output PyTree. The "leaf_filter" function exists because, at times, the user will want to specify that a whole subtree is "trainable", but the subtree may contain values that are fundamentally not trainable. One common example is a neural network which contains both trainable components (e.g. weights and biases) and non-trainable components (e.g. activations). One option is for the user to put in more work to specify "member_getter", the code would look like this:
    ```python
    pytree = {"a": 0, "b": nn}

    def member_getter(pytree):
        leaves_of_nn = jax.tree.leaves(pytree["b"])
        floats_of_nn = eqx.filter(leaves_of_nn, eqx.is_inexact_array)
        return floats_of_nn

    partition_fn = make_partition_by_members(member_getter)
    ```
    However, this is a bit more work than the user should have to do. Instead, we can provide a "leaf_filter" which filters out values from all of the leaves specified by "member_getter". Thus, the user can instead just write:

    ```python
    pytree = {"a": 0, "b": nn}

    def member_getter(pytree):
        return pytree["b"]

    partition_fn = make_partition_by_members(member_getter, leaf_filter=eqx.is_inexact_array)
    ```

    Args:
        member_getter (typing.Callable[[PyTree], PyTree]): A function that takes a PyTree and returns a PyTree of objects that specify what the first output PyTree should contain.
        leaf_filter (Callable[[typing.Any], bool], optional): A function that filters out values from the leaves specified by "member_getter". Defaults to eqx.is_inexact_array.

    Returns:
        PartitionFn: A partition function that returns (tree_with_members, tree_without_members).
    """

    if not isinstance(member_getter, Callable):
        raise TypeError("member_getter must be callable.")

    def fn(pytree: PyTree):
        # The items may be leaves or PyTree nodes. We want to get the leaves.
        items = member_getter(pytree)

        if not isinstance(items, PyTree):
            raise TypeError("Specified member_getter must return a PyTree.")

        # Filter the leaves based on the leaf_filter.
        leaves = jax.tree.leaves(items)

        leaves = eqx.filter(leaves, leaf_filter)

        targ_ids = [id(t) for t in leaves]

        def partition_spec(x):
            return id(x) in targ_ids

        trainable, static = eqx.partition(pytree, partition_spec)
        return trainable, static

    return fn
