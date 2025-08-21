import pytest
from popsim.interfaces.meq2xarray import tcv_db_to_xr, loadmat, mat_to_xr_cli
import os
from popsim import DATA_DIR
import xarray as xr
import tempfile
from click.testing import CliRunner


@pytest.fixture
def file_path():
    return os.path.join(DATA_DIR, "tcv/TCV_meqdb_template.mat")


def test_tcv_db_to_xr(file_path):
    dt = tcv_db_to_xr(file_path)
    assert isinstance(dt, xr.DataTree)
    for node in dt.descendants:
        assert node.time.size > 0

def test_loadmat(file_path):
    d = loadmat(file_path)
    assert isinstance(d, dict)


def test_mat_to_xr_cli(file_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        result = runner.invoke(mat_to_xr_cli, [
            file_path,
            temp_dir,
            "db",
        ])
        
        assert result.exit_code == 0, f"CLI failed with error: {result.output}"

        file_out = os.path.join(temp_dir, "TCV_meqdb_template.nc")
        dt = xr.open_datatree(file_out)
        assert isinstance(dt, xr.DataTree)