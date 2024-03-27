import popsim
from popsim.simulators.torax.sparc_prd import get_sim
import os
from torax import simulation_app

def test_sparc_prd():
    # Hard-coded setting of environment variables.
    # Not great, but... it works.
    os.environ["TORAX_QLKNN_MODEL_PATH"] = popsim.TORAX_QLKNN_MODEL_PATH
    PERCENT_ERROR = 10

    def within_percent_error(actual, expected):
        return abs(actual - expected) / expected < PERCENT_ERROR / 100

    # Run the simulation.
    ds = simulation_app.main(get_sim, return_xr=True)


    # Check the final state of the simulation.
    # Note that these values are not in line with the original SPARC papers.
    # Probably something off with the config/boundary conditions.
    # Worth getting a real physicist to check this.
    expected_final_ti0 = 14.0 # keV
    expected_final_te0 = 16.0 # keV
    expected_final_ne0 = 3.4 # 10^20 m^-3

    assert within_percent_error(ds["temp_ion"].isel(time=-1).sel(rho=0, method="nearest").values, expected_final_ti0)
    assert within_percent_error(ds["temp_el"].isel(time=-1).sel(rho=0, method="nearest").values, expected_final_te0)
    assert within_percent_error(ds["ne"].isel(time=-1).sel(rho=0, method="nearest").values, expected_final_ne0)