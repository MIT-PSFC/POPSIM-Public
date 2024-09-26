from popsim.xarray_utils import make_data_array, time_and_pytree_to_xarray, DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME
import numpy as np
import pytest
import xarray as xr


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
    var_test._dims = ("foo", )
    da_test = xr.DataArray(np.random.rand(10, 5), dims=(DEFAULT_TIME_DIM_NAME, "foo"), coords={"foo": np.arange(5)})
    da_test.variable._dims = ("foo",)
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
    var_test._dims = ("foo", )
    da_test = xr.DataArray(np.random.rand(3, 10, 5), dims=(DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, "foo"), coords={"foo": np.arange(5)})
    da_test.variable._dims = ("foo",)
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
