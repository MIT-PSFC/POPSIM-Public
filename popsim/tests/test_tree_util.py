import chex
import jax.numpy as jnp

from popsim import tree_util
from popsim.modules.prng import PRNGModule
from popsim.sim_utils import CombinatorialCases, MultiCases
import pytest
import xarray as xr
import jax
import equinox as eqx
import copy

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

def test_tree_transpose_docstring_example():
        # Test: List of PyTrees to PyTree of arrays
    seq_of_trees = [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}]
    tree_of_arrays = tree_util.tree_transpose(seq_of_trees)
    expected_tree_of_arrays = {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])}
    
    assert tree_of_arrays.keys() == expected_tree_of_arrays.keys()
    for key in tree_of_arrays:
        assert jnp.array_equal(tree_of_arrays[key], expected_tree_of_arrays[key])

    # Test: PyTree of arrays to list of PyTrees
    tree_of_arrays = {"a": jnp.array([0.0, 2.0]), "b": jnp.array([1.0, 3.0])}
    seq_of_trees = tree_util.tree_transpose(tree_of_arrays)
    expected_seq_of_trees = [{"a": 0.0, "b": 1.0}, {"a": 2.0, "b": 3.0}]
    
    assert len(seq_of_trees) == len(expected_seq_of_trees)
    for actual, expected in zip(seq_of_trees, expected_seq_of_trees):
        assert actual == expected

    # Test: List of xr.DataArray to xr.DataArray
    seq_of_xr = [xr.DataArray([0.0, 1.0], dims="x"), xr.DataArray([2.0, 3.0], dims="x")]
    xr_array = tree_util.tree_transpose(seq_of_xr, extra_dim_name="new_dim")
    expected_xr_array = xr.DataArray([[0.0, 1.0], [2.0, 3.0]], dims=("new_dim", "x"))
    
    xr.testing.assert_equal(xr_array, expected_xr_array)

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
    ((jax.jit(lambda x: x + 1.0), jnp.array([1.0, 2.0])), False) # Tuple with a function
])
def test_any_nans_and_no_nans(tree, has_nans):
    assert tree_util.any_nans(tree) == has_nans
    assert tree_util.no_nans(tree) == (not has_nans)


def test_tree_transpose():

    da = xr.DataArray(jnp.array([1.0, 2.0, 3.0]), dims=["x"])

    # Test case 1: List of PyTrees to PyTree of arrays
    input_tree = [{"a": 0.0, "b": 1.0, "c": da}, {"a": 2.0, "b": 3.0, "c": da}]

    input_tree = jax.tree.map(jnp.atleast_1d, input_tree)

    out = tree_util.tree_transpose(input_tree, "simulation")

    assert out["c"].equals(xr.concat([da, da], dim="simulation"))

    back = tree_util.tree_transpose(out, "simulation")

    chex.assert_trees_all_equal(input_tree, back)


def test_json_compatible():
    import json
    import tempfile

    def test_write_json(out):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as tmp:
            json.dump(out, tmp)
            tmp.seek(0)
            loaded1 = json.load(tmp)
            assert loaded1 == out

    # Test case 1: Pytree with arrays.
    tree0a = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0]), "c": {"d": jnp.array([5.0, 6.0])}}
    tree0b = tree_util.to_json_compatible(tree0a)
    assert tree0b == {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": {"d": [5.0, 6.0]}}
    test_write_json(tree0b)

    # Test case 2: things that should be left alone.
    tree1a = {"a": 1.1, "b": "foo", "c": [1, 2, 3], "d": True, "e": None}
    tree1b = tree_util.to_json_compatible(tree1a)
    assert tree1b == tree1a
    test_write_json(tree1b)

    # Test case 3: a NN.
    nn = eqx.nn.MLP(
        in_size=1,
        out_size=1,
        width_size=2,
        depth=1,
        activation=jax.nn.relu,
        final_activation=None,
        key=jax.random.PRNGKey(42),
    )
    tree2b = tree_util.to_json_compatible(nn)
    
    test_write_json(tree2b)

    assert isinstance(tree2b, dict)
    assert isinstance(tree2b['activation'], str)
    assert tree2b['final_activation'] is None

    layer0 = tree2b['layers'][0]
    layer1 = tree2b['layers'][1]

    assert layer0["bias"] == nn.layers[0].bias.tolist()
    assert layer0["weight"] == nn.layers[0].weight.tolist()

    assert layer1["bias"] == nn.layers[1].bias.tolist()
    assert layer1["weight"] == nn.layers[1].weight.tolist()
    
def test_convert_to_real():
    # Tree where some arrays have complex dtype, but all complex parts are zero.
    tree_real = {
        'a': jnp.array([1.0 + 0.0j, 3.0]),
        'b': {
            'c': 2.0,
            'd': jnp.array([5.0, 6.0+ 0.0j]),
        }
    }
    
    tree_real_out = tree_util.convert_to_real(tree_real)
    chex.assert_trees_all_close(tree_real_out, tree_real)
    
    # Tree where some arrays have non-zero complex parts.
    tree_imag = copy.deepcopy(tree_real)
    tree_imag['a'] = tree_imag['a'] + 1.0j
    # By default, should raise an error.
    with pytest.raises(ValueError):
        tree_util.convert_to_real(tree_imag)
        
    # If we disable the check, should work.
    tree_imag_out = tree_util.convert_to_real(tree_imag, check=False)
    chex.assert_trees_all_close(tree_imag_out, tree_real)
    