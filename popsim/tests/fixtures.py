import xarray as xr
import os
import pytest
import popsim


def load_cmod_test_dataset():
    path_to_data = os.path.join(popsim.DATA_DIR, "cmod/saperstein_july_2024_subset.nc")
    ds = xr.open_dataset(path_to_data)
    return ds

@pytest.fixture
def cmod_test_dataset():
    return load_cmod_test_dataset()