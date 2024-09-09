import xarray as xr
import os
import pytest
import popsim


def load_cmod_test_dataset():
    try:
        path_to_data = os.path.join(popsim.DATA_DIR, "cmod/saperstein_july_2024_subset.nc")
        ds = xr.open_dataset(path_to_data)
        return ds
    except:
        raise ValueError(f"Could not find the CMOD test dataset at {path_to_data}. Did you do a git lfs init followed by a git lfs pull?")

@pytest.fixture(scope="session")
def cmod_test_dataset():
    return load_cmod_test_dataset()