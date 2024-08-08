import equinox as eqx
from jaxtyping import PyTree

"""
Partition functions and helpers.
"""


def partition_pytree_by_members(pytree: PyTree, pytree_items: tuple[object]) -> tuple[PyTree, PyTree]:
    """Given a PyTree and a tuple of items that are members of the PyTree, partition the PyTree into two PyTrees:
    one containing the members of the PyTree that are in the tuple, and one containing the members of the PyTree that
    are not in the tuple.

    Args:
        pytree (PyTree): the PyTree to partition.
        pytree_items (tuple[object]): the items to partition the PyTree by.

    Returns:
        tuple[PyTree, PyTree]: the partitioned PyTrees. The first contains members that are in "pytree_items", and the
        second contains members that are not in "pytree_items".
    """
    targ_ids = [id(t) for t in pytree_items]

    def partition_spec(x):
        return id(x) in targ_ids

    return eqx.partition(pytree, partition_spec)
