from popsim.interfaces import build_tensorized_dataset

import xarray as xr
import numpy as np
import tempfile
import pytest
import os

def test_build_dataset():
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



def test_build_dataset_time_dim_equals_time_coord():
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