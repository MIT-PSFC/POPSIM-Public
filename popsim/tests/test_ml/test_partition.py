import jax
import jax.numpy as jnp
from popsim.ml.partition import make_partition_by_members, partition_by_arraylike
from jaxtyping import Array
import chex
import equinox as eqx

@chex.dataclass
class DummyClass:
    x: Array
    y: dict[str, Array]


def test_make_partition_by_members():
    #
    # Example 1: partitioning a simple PyTree with a NN.
    #
    nn = eqx.nn.MLP(
        in_size=2,
        out_size=2,
        width_size=32,
        depth=2,
        key=jax.random.PRNGKey(0),
    )
    tree0 = DummyClass(x=jnp.zeros(3), y={"a": jnp.arange(10), "b": nn})

    def items_to_partition_by(pytree):
        return (pytree.x, pytree.y["b"])

    partition_fn = make_partition_by_members(items_to_partition_by)

    tree0a, tree0b = partition_fn(tree0)

    # Check the partitioned trees are as expected.
    expected_nn0a, expected_nn0b = eqx.partition(nn, eqx.is_inexact_array_like)
    expected_tree0a = DummyClass(x=tree0.x, y={"a": None, "b": expected_nn0a})
    expected_tree0b = DummyClass(x=None, y={"a": tree0.y["a"], "b": expected_nn0b})
    chex.assert_trees_all_equal(tree0a, expected_tree0a)
    chex.assert_trees_all_equal(tree0b, expected_tree0b)

    # Check that the first tree has leaves that are all inexact array-like.
    leaves0a = jax.tree.leaves(tree0a)
    assert all(jax.tree.map(eqx.is_inexact_array_like, leaves0a))

    #
    # Example 2: change the tree, and test that the partition function still works.
    #
    tree1 = DummyClass(x=jnp.ones(3), y={"a": jnp.arange(10), "b": jnp.ones(5)})
    tree1a, tree1b = partition_fn(tree1)
    expected_tree1a = DummyClass(x=tree1.x, y={"a": None, "b": tree1.y["b"]})
    expected_tree1b = DummyClass(x=None, y={"a": tree1.y["a"], "b": None})
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