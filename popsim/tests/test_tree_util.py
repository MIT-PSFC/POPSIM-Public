import chex
import jax.numpy as jnp

from popsim import tree_util
from popsim.modules.prng import PRNGModule
from popsim.sim_utils import CombinatorialCases, MultiCases
import pytest
import xarray as xr
import jax


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


@pytest.mark.parametrize("input_tree, expected_output", [
    # Test case 1: List of PyTrees to PyTree of arrays
    (
        [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}],
        {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])}
    ),
    # Test case 2: PyTree of arrays to list of PyTrees
    (
        {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])},
        [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}]
    ),
    # Test case 3: Empty list
    ([], {}),
    # Test case 4: Single element list
    ([{"a": 1.0, "b": 2.0}], {"a": jnp.array([1.0]), "b": jnp.array([2.0])}),
    # Test case 5: Nested PyTree
    (
        [{"a": {"x": 1.0, "y": 2.0}, "b": 3.0}, {"a": {"x": 4.0, "y": 5.0}, "b": 6.0}],
        {"a": {"x": jnp.array([1.0, 4.0]), "y": jnp.array([2.0, 5.0])}, "b": jnp.array([3.0, 6.0])}
    ),
    # Test case 6: PyTree with different array shapes
    (
        {"a": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([5.0, 6.0])},
        [{"a": jnp.array([1.0, 2.0]), "b": 5.0}, {"a": jnp.array([3.0, 4.0]), "b": 6.0}]
    ),
    # Test case 7: PyTree with empty arrays
    (
        {"a": jnp.array([]), "b": jnp.array([])},
        [{"a": jnp.array([]), "b": jnp.array([])}]
    ),
])
def test_tree_transpose(input_tree, expected_output):
    transposed = tree_util.tree_transpose(input_tree)
    chex.assert_trees_all_close(transposed, expected_output)

    # Expect that if we transpose twice, we get back the original tree.
    double_transposed = tree_util.tree_transpose(transposed)
    chex.assert_trees_all_close(double_transposed, input_tree)

def test_tree_transpose_invalid_input():
    with pytest.raises(ValueError):
        tree_util.tree_transpose({"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0, 5.0])})
    
    with pytest.raises(ValueError):
        tree_util.tree_transpose({"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0]), "c": xr.Variable("foo", [1, 2, 3])})

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

@chex.dataclass
class EmptyDataclass:
    pass

@pytest.mark.parametrize("tree, has_nans", [
    ({"a": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([4.0, 5.0, 6.0])}, False),
    ({"a": jnp.array([1.0, jnp.nan, 3.0]), "b": jnp.array([4.0, 5.0, 6.0])}, True),
    (jnp.array([1.0, 2.0, 3.0]), False),
    (jnp.array([1.0, jnp.nan, 3.0]), True),
    (jnp.array([]), False),  # Empty array
    ({}, False),  # Empty dict
    ((), False),  # Empty tuple
    (EmptyDataclass(), False),  # Empty dataclass
])
def test_any_nans_and_no_nans(tree, has_nans):
    assert tree_util.any_nans(tree) == has_nans
    assert tree_util.no_nans(tree) == (not has_nans)