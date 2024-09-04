import typing

import equinox as eqx
from jaxtyping import PyTree

"""
Partition functions and helpers.
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


def make_partition_pytree_by_members(pytree_item_getter: typing.Callable[[PyTree], tuple[object]]) -> PartitionFn:
    """Generate a partition function that partitions a PyTree into two PyTrees based on a set of members of the PyTree.
    The set of members is specified by a "pytree_item_getter" function that takes a PyTree and returns a tuple of
    objects that are members of the PyTree. The partition function will return two PyTrees: one containing the members
    specified by the "pytree_item_getter" function and the other containing the remaining members of the PyTree.

    Args:
        pytree_item_getter (typing.Callable[[PyTree], tuple[object]]): A function that takes a PyTree and returns a tuple of objects that are members of the PyTree.

    Returns:
        PartitionFn: A function that partitions a PyTree into two PyTrees based on the set of members specified by the "pytree_item_getter" function.
    """

    def partition_fn(pytree: PyTree):
        targ_ids = [id(t) for t in pytree_item_getter(pytree)]

        def partition_spec(x):
            return id(x) in targ_ids

        return eqx.partition(pytree, partition_spec)

    return partition_fn
