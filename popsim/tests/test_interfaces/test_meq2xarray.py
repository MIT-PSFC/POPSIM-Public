import pytest
from popsim.interfaces.meq2xarray import tcv_db_to_xr, loadmat, db_mat_to_xr_cli
import os
from popsim import DATA_DIR
import xarray as xr
import tempfile


@pytest.fixture
def file_path():
    return os.path.join(DATA_DIR, "TCV_meqdb_template.mat")


def test_tcv_db_to_xr(file_path):
    dt = tcv_db_to_xr(file_path)
    assert isinstance(dt, xr.DataTree)
    for node in dt.descendants:
        assert node.time.size > 0

def test_loadmat(file_path):
    d = loadmat(file_path)
    assert isinstance(d, dict)