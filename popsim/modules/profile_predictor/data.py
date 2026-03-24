import os

import numpy as np
import xarray as xr

from popsim import DATA_DIR
from popsim.data import get_path_to_ml_data_dump


def get_ds(ds_str: str, debug: bool = False):
    if ds_str == "dummy":
        ds, episode_coord = get_dummy_torax_data(debug)
    elif ds_str == "tcv":
        ds, episode_coord = get_tcv_data(debug)
    else:
        raise ValueError(f"Unknown dataset {ds_str}")
    return ds, episode_coord


def get_dummy_torax_data(debug: bool = False):
    ds = xr.open_dataset(os.path.join(DATA_DIR, "dummy/torax_profile_predictor.nc"))
    if debug:
        ds = ds.isel(simulation=slice(0, 50))

    ds = ds.rename(
        {
            "ne": "ne20_rho",
            "temp_el": "Te_keV_rho",
            "rho_cell_norm": "rho",
        }
    )

    ds["ne20_edge"] = ds["ne20_rho"].sel(rho=1.0, method="nearest")  # 1e20 m^-3

    ds["ne20_line_avg"] = ds["ne20_rho"].integrate("rho")  # 1e20 m^-3

    # Compute means and shapes.
    ds["Te_keV_line_avg"] = ds["Te_keV_rho"].integrate("rho")
    ds["Te_shape"] = ds["Te_keV_rho"] / ds["Te_keV_line_avg"]
    ds["ne_shape"] = ds["ne20_rho"] / ds["ne20_line_avg"]

    episode_coord = "simulation"
    return ds, episode_coord


def get_tcv_data(debug: bool = False):
    path = get_path_to_ml_data_dump() / "TCV/xarrays/datatree.nc"

    dt = xr.open_datatree(path)
    if debug:
        dt = dt.isel(shot=slice(0, 50))

    # Get shots for which both thomson and fast data are available and downselect to recent shots
    shots = [s for s in dt["thomson"].shot.values if s in dt["fast"].shot.values]
    dt = dt.sel(shot=shots)

    fast_samps_at_thomson = dt["fast"].sel(time=dt["thomson"]["time"])
    ds = xr.merge([dt["thomson"].ds, fast_samps_at_thomson.ds])

    ds["Paux"] = ds["NBI"] + ds["ECRH"]

    ds["Te_keV_rho"] = 1e-3 * ds["Te_rho"]
    ds["ne20_rho"] = 1e-20 * ds["Ne_rho"]

    ds["ne20_line_avg"] = 1e-20 * ds["NEavg"]
    ds["ne20_edge"] = ds["ne20_rho"].sel(rho=1.0, method="nearest")  # 1e20 m^-3
    ds["Ip_MA"] = 1e-6 * ds["IP"]
    ds["Wtot_MJ"] = 1e-6 * ds["Wtot"]
    ds["R0"] = ds["RMAG"]

    # Rename variables to something more standard.
    ds = ds.rename({"BZERO": "B0", "KAPPA": "kappa", "Paux": "Paux_MW", "DELTA": "delta"})

    ds["B0"] = np.abs(ds["B0"])  # Make sure B0 is positive

    """Reject unrealistic data points and data points at low temperature and density, when Thomson scattering can be unreliable"""
    # Only keep data where NEavg is above 0.1e20
    ds["ne20_line_avg"] = xr.where(ds["ne20_line_avg"] > 0.1, ds["ne20_line_avg"], np.nan)

    # Only keep data where Wtot is above 1e3 J
    ds["Wtot_MJ"] = xr.where(ds["Wtot_MJ"] > 1e-3, ds["Wtot_MJ"], np.nan)

    # Only keep data where Te_rho at rho=1.0 is less than 0.25 keV and above 0.0
    ds["Te_keV_rho"] = xr.where((ds["Te_keV_rho"].sel(rho=1.0) < 0.25) & (ds["Te_keV_rho"].sel(rho=1.0) > 0.0), ds["Te_keV_rho"], np.nan)

    # # Only keep data where ne20_rho at rho=1.0 is less than 0.4e20 and above 0..0
    ds["ne20_rho"] = xr.where((ds["ne20_rho"].sel(rho=1.0) < 0.4) & (ds["ne20_rho"].sel(rho=1.0) > 0.0), ds["ne20_rho"], np.nan)

    ds["Te_shape"] = ds["Te_keV_rho"] / ds["Te_keV_rho"].integrate("rho")
    ds["ne_shape"] = ds["ne20_rho"] / ds["ne20_rho"].integrate("rho")

    episode_coord = "shot"
    return ds, episode_coord
