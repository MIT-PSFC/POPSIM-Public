import jax.numpy as jnp
from popsim.ml.partition import make_partition_by_members, partition_by_arraylike
from jaxtyping import Array
import chex

@chex.dataclass
class DummyClass:
    x: Array
    y: dict[str, Array]


def test_make_partition_by_members():
    # Test with a simple PyTree that contains a single array and a dictionary of arrays
    # We want a partition of tree0 where the first partition contains tree0.x and tree0.y["a"].
    tree0 = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": jnp.ones(5)})


    def items_to_partition_by(pytree):
        return (pytree.x, pytree.y["a"])

    expected_tree0a = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": None})
    expected_tree0b = DummyClass(x=None, y={"a": None, "b": jnp.ones(5)})

    partition_fn = make_partition_by_members(items_to_partition_by)

    tree0a, tree0b = partition_fn(tree0)
    chex.assert_trees_all_equal(tree0a, expected_tree0a)
    chex.assert_trees_all_equal(tree0b, expected_tree0b)

    # Now change the tree, and test that the partition function still works.
    tree1 = DummyClass(x=jnp.ones(3), y={"a": jnp.arange(10), "b": jnp.ones(5)})
    tree1a, tree1b = partition_fn(tree1)
    expected_tree1a = DummyClass(x=jnp.ones(3), y={"a": jnp.arange(10), "b": None})
    expected_tree1b = DummyClass(x=None, y={"a": None, "b": jnp.ones(5)})
    chex.assert_trees_all_equal(tree1a, expected_tree1a)
    chex.assert_trees_all_equal(tree1b, expected_tree1b)



def test_partition_by_arraylike():
    # Create a PyTree with array-like and non-array-like elements
    tree = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": "non-array"})

    # Expected partitioned PyTrees
    expected_arraylike_tree = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": None})
    expected_non_arraylike_tree = DummyClass(x=None, y={"a": None, "b": "non-array"})

    # Partition the PyTree
    arraylike_tree, non_arraylike_tree = partition_by_arraylike(tree)

    # Assert that the partitioned PyTrees match the expected results
    chex.assert_trees_all_equal(arraylike_tree, expected_arraylike_tree)
    chex.assert_trees_all_equal(non_arraylike_tree, expected_non_arraylike_tree)