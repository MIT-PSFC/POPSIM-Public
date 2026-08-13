import chex
import equinox as eqx
import jax
import numpy as np
import pytest
import xarray as xr

from popsim.tree_util import tree_transpose
from popsim.xarray_utils import (
    DEFAULT_SIM_DIM_NAME,
    DEFAULT_TIME_DIM_NAME,
    add_dim_to_vars,
    make_data_array,
    pytree_to_xarray,
    remove_dim_from_vars,
    run_function_with_dim_removed,
    scramble_xr,
    time_and_pytree_to_xarray,
)


@pytest.mark.parametrize("arr, expected_dims, expected_output, expect_warning", [
    (
        # 2D array with simulation and time coordinates
        np.ones((2, 3)),
        (DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME),
        lambda out: np.array_equal(out.values, np.ones((2, 3))),
        False
    ),
    (
        # 3D array with simulation and time coordinates and no extra coords
        np.ones((2, 3, 4)),
        (DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "test_extra_dim_0"),
        lambda out: np.array_equal(out.values, np.ones((2, 3, 4))),
        False
    ),
    (
        # 1D array, expect warning
        np.ones(2),
        None,
        lambda out: out is None,
        True
    ),
])
def test_make_data_array(arr, expected_dims, expected_output, expect_warning):
    nsim = 2
    ntime = 3
    coords = {DEFAULT_SIM_DIM_NAME: xr.DataArray(np.arange(nsim), dims=(DEFAULT_SIM_DIM_NAME, )), DEFAULT_TIME_DIM_NAME: xr.DataArray(np.arange(ntime), dims=(DEFAULT_TIME_DIM_NAME, ))}


    if expect_warning:
        with pytest.warns(UserWarning):
            out = make_data_array("test", arr, dims=[DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME], coords = coords)
    else:
        out = make_data_array("test", arr, dims=[DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME], coords = coords)

    if expected_dims is not None:
        assert out.dims == expected_dims
    
    assert expected_output(out)

# Helper function to create a simple tree
def create_tree(shape):
    return {
        'a': np.random.rand(*shape),
        'b': {
            'c': np.random.rand(*shape),
            'd': np.random.rand(*shape)
        }
    }


"""
Tests for time_and_pytree_to_xarray function.
"""
# Case 1: Single simulation, 1D time array
def test_single_simulation_1d_time():
    time = np.arange(10)
    tree = create_tree((10,))
    result = time_and_pytree_to_xarray(time, tree)
    
    assert isinstance(result, xr.Dataset)
    assert set(result.dims) == {'time'}
    assert result.sizes['time'] == 10
    assert 'simulation' not in result.dims
    assert set(result.data_vars) == {'a', 'b.c', 'b.d'}

# Case 2: Single simulation, 2D time array (expect failure)
def test_single_simulation_2d_time():
    time = np.arange(20).reshape(2, 10)
    tree = create_tree((10,))
    
    with pytest.raises(AssertionError):
        time_and_pytree_to_xarray(time, tree)

# Case 3: Multi simulation, 1D time array
def test_multi_simulation_1d_time():
    time = np.arange(10)
    tree = create_tree((3, 10))
    result = time_and_pytree_to_xarray(time, tree, multi_simulation=True)
    
    assert isinstance(result, xr.Dataset)
    assert set(result.dims) == {'simulation', 'time'}
    assert result.sizes['time'] == 10
    assert result.sizes['simulation'] == 3
    assert set(result.data_vars) == {'a', 'b.c', 'b.d'}

# Case 4: Multi simulation, 2D time array
def test_multi_simulation_2d_time():
    time = np.arange(30).reshape(3, 10)
    tree = create_tree((3, 10))
    result = time_and_pytree_to_xarray(time, tree, multi_simulation=True)
    
    assert isinstance(result, xr.Dataset)
    assert set(result.dims) == {'simulation', 'time'}
    assert result.sizes['time'] == 10
    assert result.sizes['simulation'] == 3
    assert set(result.data_vars) == {'a', 'b.c', 'b.d'}
    np.testing.assert_array_equal(result.time.values, time)

# Case 5: Multi simulation, 2D time array with all the same time values
def test_multi_simulation_2d_time_same_values():
    time = np.tile(np.arange(10), (3, 1))
    tree = create_tree((3, 10))
    result = time_and_pytree_to_xarray(time, tree, multi_simulation=True)
    
    assert isinstance(result, xr.Dataset)
    assert set(result.dims) == {'simulation', 'time'}
    assert result.sizes['time'] == 10
    assert result.sizes['simulation'] == 3
    assert set(result.data_vars) == {'a', 'b.c', 'b.d'}
    np.testing.assert_array_equal(result.time.values, np.arange(10))

# Case 6: Single simulation, 1D time array with xr.Variable and xr.DataArray in the tree
def test_single_simulation_1d_time_xr():
    time = np.arange(10)
    time_da = xr.DataArray(time, dims=(DEFAULT_TIME_DIM_NAME,))
    tree = create_tree((10,))

    # We need to test a xr.Variable that has dimensions (time, spatial) in the array
    # but not in the tree to emulate what happens when we simulate. Similarly for xr.DataArray.
    var_test = xr.Variable(data=np.random.rand(10, 5), dims=(DEFAULT_TIME_DIM_NAME, "foo"))
    da_test = xr.DataArray(np.random.rand(10, 5), dims=(DEFAULT_TIME_DIM_NAME, "foo"), coords={"foo": np.arange(5)})
    tree["var"] = var_test
    tree["da"] = da_test

    result = time_and_pytree_to_xarray(time, tree)
    assert isinstance(result, xr.Dataset)
    assert set(result.dims) == {'time', 'foo'}
    assert result.sizes['time'] == 10
    assert 'simulation' not in result.dims
    assert set(result.data_vars) == {'a', 'b.c', 'b.d', 'var', 'da'}
    assert result["var"].dims == (DEFAULT_TIME_DIM_NAME, "foo")
    assert result["da"].dims == (DEFAULT_TIME_DIM_NAME, "foo")
    assert set(result.coords.keys()) == {DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "foo"}

# Case 7: like case 6, but multi-simulation.
def test_multi_simulation_2d_time_xr():
    time = np.arange(30).reshape(3, 10)
    tree = create_tree((3, 10))

    # We need to test a xr.Variable that has dimensions (time, spatial) in the array
    # but not in the tree to emulate what happens when we simulate. Similarly for xr.DataArray.
    var_test = xr.Variable(data=np.random.rand(3, 10, 5), dims=('simulation', 'time', 'foo'))
    da_test = xr.DataArray(np.random.rand(3, 10, 5), dims=(DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "foo"), coords={"foo": np.arange(5)})
    tree["var"] = var_test
    tree["da"] = da_test
    
    result = time_and_pytree_to_xarray(time, tree, multi_simulation=True)


    assert isinstance(result, xr.Dataset)
    assert set(result.dims) == {'simulation', 'time', "foo"}
    assert set(result.coords.keys()) == {DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "foo"}
    assert result.sizes['time'] == 10
    assert result.sizes['simulation'] == 3
    assert set(result.data_vars) == {'a', 'b.c', 'b.d', "var", "da"}
    assert result["var"].dims == (DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "foo")
    assert result["da"].dims == (DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "foo")

def test_run_function_with_dim_removed():
    var = xr.Variable(data=np.random.rand(3, 10, 5), dims=('simulation', 'time', 'foo'))
    remove_dim = 'simulation'

    @jax.jit
    def fn(v):
        assert 'simulation' not in v.dims
        assert 'time' in v.dims
        assert 'foo' in v.dims
        return v

    res = run_function_with_dim_removed(fn, (var,), remove_dim, in_axes=0)
    assert res.dims == var.dims
    np.testing.assert_allclose(np.asarray(res.data), np.asarray(var.data))


# Sample test data
def sample_tree():
    """Creates a PyTree with xr.Variable instances for testing."""
    var1 = xr.Variable(('x',), [1, 2, 3])
    var2 = xr.Variable(('y',), [4, 5, 6, 7])
    return {'a': var1, 'b': {'c': var2}}


def test_add_dim_to_vars():
    tree = sample_tree()
    dim_name = 'time'

    # Apply add_dim_to_vars
    modified_tree = add_dim_to_vars(tree, dim_name)

    # Check if the dimension was added to each xr.Variable
    assert modified_tree['a'].dims == ('time', 'x')
    assert modified_tree['b']['c'].dims == ('time', 'y')


def test_remove_dim_from_vars():
    tree = sample_tree()
    dim_name = 'time'

    # First add a dimension so it can be removed
    tree_with_added_dim = add_dim_to_vars(tree, dim_name)

    # Apply remove_dim_from_vars
    modified_tree = remove_dim_from_vars(tree_with_added_dim, dim_name)

    # Check if the dimension was removed from each xr.Variable
    assert modified_tree['a'].dims == ('x',)
    assert modified_tree['b']['c'].dims == ('y',)


def test_add_then_remove_dim():
    tree = sample_tree()
    dim_name = 'time'

    # Apply add_dim_to_vars and then remove_dim_from_vars
    modified_tree = add_dim_to_vars(tree, dim_name)
    restored_tree = remove_dim_from_vars(modified_tree, dim_name)

    chex.assert_trees_all_close(tree, restored_tree)


def test_add_remove_nonexistent_dim():
    tree = sample_tree()
    dim_name = 'nonexistent'

    # Apply remove_dim_from_vars without adding the dimension first
    modified_tree = remove_dim_from_vars(tree, dim_name)

    chex.assert_trees_all_equal(tree, modified_tree)

@pytest.mark.parametrize("zero", [True, False])
def test_scramble_xr_da(zero):
    #
    # Test scramble for xr.DataArray.
    #
    da = xr.DataArray(np.random.rand(3, 4), dims=["x", "y"], coords={"x": np.arange(3), "y": np.arange(4)})
    scrambled_da = scramble_xr(da, zero=zero)
    
    if zero:
        assert (scrambled_da.values == 0.0).all()
    else:
        # Check scrambled_da does not have same data as da.
        assert not np.array_equal(scrambled_da.values, da.values)
    
    # Check everything else is the same.
    assert scrambled_da.dims == da.dims
    assert scrambled_da.coords.equals(da.coords)
    
    #
    # Test scramble for xr.Dataset.
    #
    ds = xr.Dataset({"var1": da, "var2": da})
    scrambled_ds = scramble_xr(ds, zero=zero)
    for var in ds.data_vars:
        if zero:
            assert (scrambled_ds[var].values == 0.0).all()
        else:
            assert not np.array_equal(scrambled_ds[var].values, ds[var].values)
        assert scrambled_ds[var].dims == ds[var].dims
        assert scrambled_ds[var].coords.equals(ds[var].coords)
    assert set(scrambled_ds.data_vars) == set(ds.data_vars)
    
    #
    # Test scramble for xr.DataTree.
    #
    dt = xr.DataTree.from_dict({"node1": ds, "node2": ds})
    scrambled_dt = scramble_xr(dt, zero=zero)
    for node in ["node1", "node2"]:
        scrambled_dt_ds = scrambled_dt[node].ds
        dt_ds = dt[node].ds
        assert isinstance(scrambled_dt_ds, xr.Dataset)
        for var in scrambled_dt_ds.data_vars:
            if zero:
                assert (scrambled_dt_ds[var].values == 0.0).all()
            else:
                assert not np.array_equal(scrambled_dt_ds[var].values, dt_ds[var].values)
            assert scrambled_dt_ds[var].dims == dt_ds[var].dims
            assert scrambled_dt_ds[var].coords.equals(dt_ds[var].coords)
    