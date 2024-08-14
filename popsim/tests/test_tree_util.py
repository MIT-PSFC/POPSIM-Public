import chex
import jax.numpy as jnp

from popsim import tree_util
from popsim.modules.prng import PRNGModule
from popsim.sim_utils import CombinatorialCases, MultiCases
import pytest

# Sample data structures for testing
sample_tree_1 = {
    'a': [1, 2, 3],
    'b': {'c': 'hello', 'd': [4.5, 6.7], 'comb_cases': CombinatorialCases(cases={"a": 4, "b": [5, 6]})},
    'e': 42
}

sample_tree_2 = [
    [1, 'two', 3.0],
    {'a': 4, 'b': '5', 'c': [6, '7']},
    8,
    MultiCases(cases={"foo": [1, 2, 3], "bar": PRNGModule.State(seed=42)}),
]

@pytest.mark.parametrize("tree, type_, expected", [
    (sample_tree_1, int, [1, 2, 3, 4, 5, 6, 42]),
    (sample_tree_1, float, [4.5, 6.7]),
    (sample_tree_1, str, ['hello']),
    (sample_tree_1, CombinatorialCases, [sample_tree_1["b"]["comb_cases"]]),
    (sample_tree_1, complex, []),
    (sample_tree_2, MultiCases, [sample_tree_2[3]]),
    (sample_tree_2, PRNGModule.State, [sample_tree_2[3].cases["bar"]]),
])
def test_get_instances_from_tree_leaves(tree, type_, expected):
    result = tree_util.get_instances_from_tree_leaves(tree, type_)
    assert result == expected



def test_tree_transpose():
    def make_tree(x):
        return {"foo": {"a": x + 1.0, "b": x + 2.0}, "bar": x + 3.0}

    seq_of_pytrees = [make_tree(x) for x in range(3)]
    pytree_of_arrays = tree_util.tree_transpose(seq_of_pytrees)

    assert jnp.all(pytree_of_arrays["foo"]["a"] == jnp.array([1.0, 2.0, 3.0]))
    assert jnp.all(pytree_of_arrays["foo"]["b"] == jnp.array([2.0, 3.0, 4.0]))
    assert jnp.all(pytree_of_arrays["bar"] == jnp.array([3.0, 4.0, 5.0]))


def test_leaves_as_array():
    labels = ["b", "a", "c"]
    values = jnp.array([1.0, 2.0, 3.0])
    ordered_dict = tree_util.build_ordered_dict(labels, values)
    assert ordered_dict == {"b": 1.0, "a": 2.0, "c": 3.0}

    # Check that we can reconstruct the original array.
    array = tree_util.leaves_as_array(ordered_dict)
    chex.assert_trees_all_close(array, values)

    # Check that keys are indeed ordered.
    assert list(ordered_dict.keys()) == labels
