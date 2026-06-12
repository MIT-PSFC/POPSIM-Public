from popsim.data.dataset_utils import build_tensorized_dataset, add_to_zarr_store
from popsim.tests.fixtures import tcv_fbt_test_dataset
from popsim.data.data_generators import dummy
import xarray as xr
import numpy as np
import tempfile
import pytest
import os
import shutil
import pytest
import sys

# Generation of test examples uses PRNG. We found failure modes by repeating tests at some point, so
# lets make repeats of tests a part of testing.
N_TEST_REPEAT = 3


@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_add_to_zarr_store(test_number):
    np.random.seed(test_number)
    dummy_generator = dummy.generate_simple_scalar_dataset(3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # There should be no zarr store initially
        assert not os.path.exists(zarr_path)
        
        # Add datasets to zarr store
        for ds in dummy_generator:
            success = add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode")
            assert success
            assert os.path.exists(zarr_path)

        # Open the zarr store and check its contents.
        ds_store = xr.open_zarr(zarr_path, consolidated=False).load()
        expected_episodes = xr.DataArray([0, 1, 2], dims=["episode"], coords={"episode": [0, 1, 2]})
        assert ds_store["episode"].equals(expected_episodes)
        
        # Check that the "var" variable has the expected values.
        expected_var = xr.DataArray(np.array([[0.0, np.nan, np.nan],
                                              [0.0, 1.0, np.nan],
                                              [0.0, 1.0, 2.0]]),
                                      dims=["episode", "time_idx"],
                                      coords = {"episode": ("episode", [0, 1, 2])})
        assert ds_store["var"].equals(expected_var)

        # Expect that time is a variable.
        expected_time = xr.DataArray(np.array([[0.0, np.nan, np.nan],
                                                  [0.0, 1.0, np.nan],
                                                    [0.0, 1.0, 2.0]]),
                                        dims=["episode", "time_idx"],
                                        coords={"episode": ("episode", [0, 1, 2])})
        assert ds_store["time"].equals(expected_time)

@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_add_to_zarr_store_changing_spatial(test_number):
    np.random.seed(test_number)
    dummy_generator = dummy.generate_dataset_with_changing_spatial_var(3)

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # There should be no zarr store initially
        assert not os.path.exists(zarr_path)
        
        # Add datasets to zarr store
        for ds in dummy_generator:
            success = add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode")
            assert success
            assert os.path.exists(zarr_path)

        # Open the zarr store and check its contents.
        ds_store = xr.open_zarr(zarr_path, consolidated=False).load()
        expected_episodes = xr.DataArray([0, 1, 2], dims=["episode"], coords={"episode": [0, 1, 2]})
        assert ds_store["episode"].equals(expected_episodes)
        
        # Check that the "var" variable has the expected values.
        expected_var = xr.DataArray(
            np.array([
                [   # Episode 0
                    [0.0, np.nan, np.nan],  # Timestep 0
                    [0.0, np.nan, np.nan],  # Timestep 1
                    [0.0, np.nan, np.nan],  # Timestep 2
                ],
                [   # Episode 1
                    [0.0, 1.0, np.nan],
                    [0.0, 1.0, np.nan],
                    [0.0, 1.0, np.nan],
                ],
                [   # Episode 2
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ],
            ]),
            dims=["episode", "time_idx", "space_idx"],
            coords={"episode": ("episode", [0, 1, 2])}
        )
        assert ds_store["var"].equals(expected_var)

        # Expect that space is a variable.
        expected_space = xr.DataArray(np.array([[0.0, np.nan, np.nan],
                                                  [0.0, 1.0, np.nan],
                                                    [0.0, 1.0, 2.0]]),
                                        dims=["episode", "space_idx"],
                                        coords={"episode": ("episode", [0, 1, 2])})
        assert ds_store["space"].equals(expected_space)

@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_add_to_zarr_store_mismatch_dims(test_number):
    """Test trying to add a dataset with different dimensions to the zarr store raises an error"""
    np.random.seed(test_number)
    dummy_generator = dummy.generate_simple_scalar_dataset(2)

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # There should be no zarr store initially
        assert not os.path.exists(zarr_path)
        
        # Add datasets to zarr store
        for ds in dummy_generator:
            success = add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode")
            assert success
            assert os.path.exists(zarr_path)

        # Add a dataset with a new dimension
        data = np.tile(np.arange(3, dtype=float), (3, 1))
        ds = xr.Dataset(
            data_vars={
                "new_var": (("time_idx", "space_idx"), data),
                "var": (("time_idx"), np.arange(3, dtype=float)),
            },
            coords={
                "time": ("time_idx", np.arange(3, dtype=float)),
                "space": ("space_idx", np.arange(3, dtype=float)),
                "episode": ("episode", [2])
            }
        )
        # Ensure this throws a value error
        with pytest.raises(ValueError):
            add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode")

@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_build_tensorized_dataset(test_number):
    np.random.seed(test_number)
    def build_fn(path: str) -> xr.Dataset:
        # Create some random data for testing
        nt = np.random.randint(1, 50)
        data = np.random.rand(nt, 5)
        time_series = np.random.rand(nt)
        times = np.random.choice(np.arange(100), size=nt, replace=False)
        times.sort()
        ds = xr.Dataset(
            {
                "data": (("time_idx", "space"), data),
                "time_series": (("time_idx"), time_series)
            },
            coords={
                "time": ("time_idx", times),
                "space": ("space", np.arange(5)),
                "shot": path
            }
        )
        return ds
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # Build the dataset
        ds = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=["test1", "test2", "test3"],
            time_dim="time_idx",
            episode_dim="shot"
        )
        
        # Check if the dataset is created correctly
        assert isinstance(ds, xr.Dataset)
        assert "data" in ds.data_vars
        assert "shot" in ds.coords
        assert "space" in ds.coords
        assert "time" in ds.data_vars
        assert "time_series" in ds.data_vars
        assert ds["time"].dims == ("shot", "time_idx")
        assert ds["time_idx"].dims == ("time_idx",)
        assert ds.sizes["shot"] == 3
        assert os.path.exists(zarr_path)

        # Try to extend the zarr store without extend_existing flag, expect it to fail.
        with pytest.raises(ValueError):
            ds = build_tensorized_dataset(
                process_fn=build_fn,
                zarr_path=zarr_path,
                identifiers=["test4"],
                time_dim="time_idx",
                episode_dim="shot",
                extend_existing=False
            )
        
        # Now try to extend the dataset with the extend_existing flag.
        ds = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=["test4"],
            time_dim="time_idx",
            episode_dim="shot",
            extend_existing=True
        )
        assert ds.sizes["shot"] == 4
        
        # Delete the zarr store and try with episodes_per_chunk instead of mb_per_chunk.
        shutil.rmtree(zarr_path)
        ds = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=["test1", "test2", "test3"],
            time_dim="time_idx",
            episode_dim="shot",
            episodes_per_chunk=2,
            mb_per_chunk=None
        )
        assert ds.chunks["shot"] == (2, 1)
        
        # Make sure that all dimensions except shot are covered by a single chunk.
        for dim in ds.dims:
            if dim != "shot":
                assert ds.chunksizes[dim] == (ds.sizes[dim],)
        
        # By default, mb_per_chunk is specified and we should get an error if the user tries to set both.
        with pytest.raises(ValueError):
            ds = build_tensorized_dataset(
                process_fn=build_fn,
                zarr_path=zarr_path,
                identifiers=["test1", "test2", "test3"],
                time_dim="time_idx",
                episode_dim="shot",
                episodes_per_chunk=2,
                mb_per_chunk=10
            )

@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_build_tensorized_dataset_time_dim_equals_time_coord(test_number):
    np.random.seed(test_number)
    def build_fn(path: str) -> xr.Dataset:
        # Create some random data for testing
        nt = np.random.randint(1, 50)
        data = np.random.rand(nt, 5)
        time_series = np.random.rand(nt)
        times = np.random.choice(np.arange(100), size=nt, replace=False)
        times.sort()
        ds = xr.Dataset(
            {
                "data": (("time_idx", "space"), data),
                "time_series": (("time_idx"), time_series)
            },
            coords={
                "time": ("time_idx", times),
                "space": ("space", np.arange(5)),
                "shot": path
            }
        )
        return ds
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # Build the dataset
        ds = build_tensorized_dataset(
            process_fn=build_fn,
            identifiers=["test1", "test2", "test3"],
            zarr_path=zarr_path,
            time_dim="time_idx",
            episode_dim="shot"
        )
        assert ds["time"].dims == ("shot", "time_idx")
        # Check that time across shots is not constant
        assert np.abs(ds["time"].diff("shot")).max().compute() > 0

@pytest.mark.skipif(sys.platform == "darwin", reason="Test fails on mac for unknown reasons. https://github.com/cfs-energy-internal/POPSIM/issues/147")
@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_build_tensorized_dataset_tcv_fbt(tcv_fbt_test_dataset, test_number):
    """Test build_tensorized_dataset with the TCV FBT dataset."""
    ds = tcv_fbt_test_dataset
    np.random.seed(test_number)
    
    def build_fn(identifier: str) -> xr.Dataset:
        # Randomly select time dimension length
        nt = np.random.randint(1, ds.sizes["time_idx"])
        
        # Create subset dataset with modified shot coordinate
        subset_ds = ds.isel(time_idx=slice(0, nt)).copy()
        subset_ds["shot"] = ("shot", [identifier])
        return subset_ds
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/tcv_test_zarr_store.zarr"
        
        # Use mock identifiers for testing
        identifiers = [100000, 100001]
        
        # Build the tensorized dataset
        result_ds = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=identifiers,
            time_dim="time_idx",
            episode_dim="shot"
        )
        # Basic checks
        assert isinstance(result_ds, xr.Dataset)
        assert "shot" in result_ds.dims
        assert result_ds.sizes["shot"] == len(identifiers)
        assert os.path.exists(zarr_path)
        result_ds = result_ds.load()
        
        # Now try appending more data
        new_identifiers = [100002, 100003, 100004, 100005, 100006]
        result_ds2 = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=new_identifiers,
            time_dim="time_idx",
            episode_dim="shot",
            extend_existing=True
        )
        # Check that the new result has the right size.
        assert result_ds2.sizes["shot"] == len(identifiers) + len(new_identifiers)
        
        # Perform a check that the data in result_ds for the same shot is the same as the data in result_ds2.
        # ds2_shot may be larger because of padding in the time_idx dimension.
        ds_shot = result_ds.sel(shot=100000)
        ds2_shot = result_ds2.load().sel(shot=100000).isel(time_idx=slice(0, ds_shot.sizes["time_idx"]))
        assert ds_shot.equals(ds2_shot)

@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_add_to_zarr_store_with_dim_sizes(test_number):
    """With dim_sizes provided, every episode is padded to the bound and the store is never extended."""
    np.random.seed(test_number)
    dummy_generator = dummy.generate_simple_scalar_dataset(3)
    dim_sizes = {"time_idx": 5}

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"

        for ds in dummy_generator:
            success = add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode", dim_sizes=dim_sizes)
            assert success
            # The store should always be at the upper-bound size, even after the first episode
            ds_store = xr.open_zarr(zarr_path, consolidated=False)
            assert ds_store.sizes["time_idx"] == 5

        ds_store = xr.open_zarr(zarr_path, consolidated=False).load()
        expected_var = xr.DataArray(np.array([[0.0, np.nan, np.nan, np.nan, np.nan],
                                              [0.0, 1.0, np.nan, np.nan, np.nan],
                                              [0.0, 1.0, 2.0, np.nan, np.nan]]),
                                    dims=["episode", "time_idx"],
                                    coords={"episode": ("episode", [0, 1, 2])})
        assert ds_store["var"].equals(expected_var)


@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_add_to_zarr_store_dim_sizes_exceeded(test_number):
    """An episode larger than the dim_sizes upper bound raises an error."""
    np.random.seed(test_number)
    dummy_generator = dummy.generate_simple_scalar_dataset(3)
    dim_sizes = {"time_idx": 2}

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"

        datasets = list(dummy_generator)
        # First two episodes have 1 and 2 timesteps, within the bound
        for ds in datasets[:2]:
            assert add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode", dim_sizes=dim_sizes)
        # Third episode has 3 timesteps, exceeding the bound
        with pytest.raises(ValueError):
            add_to_zarr_store(datasets[2], zarr_path, time_dim="time_idx", episode_dim="episode", dim_sizes=dim_sizes)


@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
def test_build_tensorized_dataset_with_dim_sizes(test_number):
    """dim_sizes overshoot is trimmed away during the rechunking pass."""
    np.random.seed(test_number)
    episode_lengths = {"test1": 10, "test2": 30, "test3": 20}

    def build_fn(path: str) -> xr.Dataset:
        nt = episode_lengths[path]
        ds = xr.Dataset(
            {
                "data": (("time_idx", "space"), np.random.rand(nt, 5)),
            },
            coords={
                "time": ("time_idx", np.arange(nt, dtype=float)),
                "space": ("space", np.arange(5)),
                "shot": path
            }
        )
        return ds

    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"

        # Upper bound of 50 deliberately overshoots the largest episode (30)
        ds = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=["test1", "test2", "test3"],
            time_dim="time_idx",
            episode_dim="shot",
            dim_sizes={"time_idx": 50, "space": 5},
        )

        assert ds.sizes["shot"] == 3
        # The overshoot beyond the largest episode should have been trimmed away
        assert ds.sizes["time_idx"] == 30
        assert ds.sizes["space"] == 5

        # Data must round-trip unchanged for each episode
        ds = ds.load()
        for path, nt in episode_lengths.items():
            episode = ds.sel(shot=path)
            assert episode["data"].isel(time_idx=slice(0, nt)).notnull().all()
            assert episode["data"].isel(time_idx=slice(nt, None)).isnull().all()


@pytest.mark.parametrize('test_number', range(N_TEST_REPEAT))
@pytest.mark.parametrize('extend_existing', [True, False])
def test_build_tensorized_dataset_no_successful(test_number, extend_existing):
    np.random.seed(test_number)

    def build_fn(path: str) -> xr.Dataset:
        return None # Simulate failure to process
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # Build the dataset
        ds = build_tensorized_dataset(
            process_fn=build_fn,
            zarr_path=zarr_path,
            identifiers=["test1", "test2", "test3"],
            time_dim="time_idx",
            episode_dim="shot",
            extend_existing=extend_existing
        )
        
        # Check if the dataset is created correctly
        assert isinstance(ds, xr.Dataset)
        assert ds.data_vars == {}
        assert ds.coords == {}
        assert ds.dims == {}
        assert not os.path.exists(zarr_path)
