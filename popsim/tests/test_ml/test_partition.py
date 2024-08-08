import jax.numpy as jnp
from popsim.ml.partition import partition_pytree_by_members
from jaxtyping import Array
import chex

@chex.dataclass
class DummyClass:
    x: Array
    y: dict[str, Array]


def test_partition_pytree_by_members():
    # Test with a simple PyTree that contains a single array and a dictionary of arrays
    # We want a partition of tree0 where the first partition contains tree0.x and tree0.y["a"].
    tree0 = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": jnp.ones(5)})
    items_to_partition_by = (tree0.x, tree0.y["a"])

    expected_tree0a = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": None})
    expected_tree0b = DummyClass(x=None, y={"a": None, "b": jnp.ones(5)})

    tree0a, tree0b = partition_pytree_by_members(tree0, items_to_partition_by)
    chex.assert_trees_all_equal(tree0a, expected_tree0a)
    chex.assert_trees_all_equal(tree0b, expected_tree0b)