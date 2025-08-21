from popsim.data.dataset_utils import build_tensorized_dataset, add_to_zarr_store
from popsim.tests.fixtures import tcv_fbt_test_dataset
from popsim.data.data_generators import dummy
import xarray as xr
import numpy as np
import tempfile
import pytest
import os


def test_add_to_zarr_store():
    dummy_iterator = dummy.generate_simple_scalar_dataset(3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # There should be no zarr store initially
        assert not os.path.exists(zarr_path)
        
        # Add datasets to zarr store
        for ds in dummy_iterator:
            success = add_to_zarr_store(ds, zarr_path, time_dim="time_idx", episode_dim="episode")
            assert success
            assert os.path.exists(zarr_path)

        # Open the zarr store and check its contents.
        ds_store = xr.open_zarr(zarr_path).load()
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


def test_build_tensorized_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/test_zarr_store.zarr"
        
        # Build the dataset
        ds = build_tensorized_dataset(
            process_fn=lambda ds: ds,
            identifiers=dummy.generate_dataset_with_spatial_var(["eps1", "eps2", "eps3"]),
            zarr_path=zarr_path,
            time_dim="time_idx",
            episode_dim="episode"
        )
        
        # Check if the dataset is created correctly
        assert isinstance(ds, xr.Dataset)
        assert "data" in ds.data_vars
        assert "episode" in ds.coords
        assert "space" in ds.coords
        assert "time" in ds.data_vars
        assert "time_series" in ds.data_vars
        assert ds["time"].dims == ("episode", "time_idx")
        assert ds["time_idx"].dims == ("time_idx",)
        assert ds.sizes["episode"] == 3
        assert ds["episode"].values.tolist() == ["eps1", "eps2", "eps3"]
        assert os.path.exists(zarr_path)

        # Try to extend the zarr store without extend_existing flag, expect it to fail.
        with pytest.raises(ValueError):
            ds = build_tensorized_dataset(
                process_fn=lambda ds: ds,
                zarr_path=zarr_path,
                identifiers=dummy.generate_dataset_with_spatial_var(["eps4"]),
                time_dim="time_idx",
                episode_dim="episode",
                extend_existing=False
            )
        
        # Now try to extend the dataset with the extend_existing flag.
        ds = build_tensorized_dataset(
            process_fn=lambda ds: ds,
            zarr_path=zarr_path,
            identifiers=dummy.generate_dataset_with_spatial_var(["eps4"]),
            time_dim="time_idx",
            episode_dim="episode",
            extend_existing=True
        )
        assert ds.sizes["episode"] == 4
        assert ds["episode"].values.tolist() == ["eps1", "eps2", "eps3", "eps4"]


def test_build_tensorized_dataset_tcv_fbt(tcv_fbt_test_dataset):
    """Test build_tensorized_dataset with the TCV FBT dataset."""
    ds = tcv_fbt_test_dataset
    
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
        identifiers = [100000, 100001, 100002]
        
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
        new_identifiers = [100003, 100004]
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
        # Check that the new result first shot matches the old result.
        assert result_ds2.isel(shot=0).load().equals(result_ds.isel(shot=0))
