import xarray as xr
import os
import pytest

@pytest.fixture
def cmod_test_dataset():
    # TODO(allenw): replace with path to commited data once permissions are sorted out.
    path_to_data = os.path.join(os.environ["ML_DATA_DUMP"], "CMOD/disruptions_warning_database.nc")
    ds = xr.open_dataset(path_to_data)
    return ds