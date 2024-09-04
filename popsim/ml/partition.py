import typing

import equinox as eqx
from jaxtyping import PyTree

"""
Partition functions and helpers. A partition function is a function that takes in a PyTree and splits that PyTree into two disjoint PyTrees. That is:
    f(tree) -> (tree1, tree2)
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


def make_partition_by_members(pytree_item_getter: typing.Callable[[PyTree], tuple[object]]) -> PartitionFn:
    """Generate a partition function that partitions a PyTree into two PyTrees based on a set of members of the PyTree.
    The set of members is specified by a "pytree_item_getter" function that takes a PyTree and returns a tuple of
    objects that are members of the PyTree. The partition function will return two PyTrees: one containing the members
    specified by the "pytree_item_getter" function and the other containing the remaining members of the PyTree.

    Args:
        pytree_item_getter (typing.Callable[[PyTree], tuple[object]]): A function that takes a PyTree and returns a tuple of objects that are members of the PyTree.

    Returns:
        PartitionFn: A partition function that returns (tree_with_members, tree_without_members).
    """

    def partition_fn(pytree: PyTree):
        targ_ids = [id(t) for t in pytree_item_getter(pytree)]

        def partition_spec(x):
            return id(x) in targ_ids

        return eqx.partition(pytree, partition_spec)

    return partition_fn
