from popsim.xarray_utils import make_data_array, time_and_pytree_to_xarray
import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize("arr, expected_dims, expected_output, expect_warning", [
    (
        # 2D array with simulation and time coordinates
        np.ones((2, 3)),
        ("simulation", "time"),
        lambda out: np.array_equal(out.values, np.ones((2, 3))),
        False
    ),
    (
        # 3D array with simulation and time coordinates and no extra coords
        np.ones((2, 3, 4)),
        ("simulation", "time", "test_extra_dim_0"),
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
    coords = {"simulation": xr.DataArray(np.arange(nsim), dims=("simulation", )), "time": xr.DataArray(np.arange(ntime), dims=("time", ))}


    if expect_warning:
        with pytest.warns(UserWarning):
            out = make_data_array("test", arr, dims=["simulation", "time"], coords = coords)
    else:
        out = make_data_array("test", arr, dims=["simulation", "time"], coords = coords)

    if expected_dims is not None:
        assert out.dims == expected_dims
    
    assert expected_output(out)



def test_time_and_pytree_to_xarray():
    # Helper function to create a simple tree
    def create_tree(shape):
        return {
            'a': np.random.rand(*shape),
            'b': {
                'c': np.random.rand(*shape),
                'd': np.random.rand(*shape)
            }
        }

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

    # Run all test cases
    test_single_simulation_1d_time()
    test_single_simulation_2d_time()
    test_multi_simulation_1d_time()
    test_multi_simulation_2d_time()
    test_multi_simulation_2d_time_same_values()