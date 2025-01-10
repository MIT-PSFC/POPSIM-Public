from torax.examples.basic_config import CONFIG
from popsim.interfaces.torax import run_torax, get_sparc_lmode_base_config
import xarray as xr

def test_run_torax_single():
    # Test we can give a config dictionary and get a result.
    ds = run_torax(CONFIG)
    assert ds["sim_error"].max() == 0

    # Also test that we can run a SPARC l-mode like scenario.
    ds = run_torax(get_sparc_lmode_base_config())
    assert ds["sim_error"].max() == 0

def test_run_torax_list():
    # Test we can give a list of config dictionaries and get a result.
    ds = run_torax([CONFIG, CONFIG])
    assert ds["sim_error"].max() == 0
    assert ds.sizes["simulation"] == 2