import jax
import numpy as np
import xarray as xr

REQUIRED_SIGNALS = [
    # Target profiles
    "ne20_rho",
    "Te_keV_rho",
    # Inputs for transport predictor
    "R0",
    "B0",
    "Ip_MA",
    "a_minor",
    "kappa",
    "delta_top",
    "delta_bottom",
]

INPUT_POWER_SIGNALS = ["P_ECRH_MW", "P_NBI_MW", "P_ICRF_MW", "P_LH_MW"]


def get_ds(
    ds_path: str,
    debug: bool | None = False,
) -> tuple[xr.Dataset, str]:
    """Load the dataset, and do some light processing to get it ready for training.

    Args:
        ds_path (str): Path to the dataset.
        debug (bool, optional): Whether to enable debug mode, reducing dataset size to at most 50 shots.

    Returns:
        tuple[xr.Dataset, str]: The processed dataset and the dimension along which to group the data
    """

    # Load dataset according to JAX setting and file path
    if ds_path.endswith(".zarr"):
        ds = xr.open_zarr(ds_path).astype(jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32)
    else:
        ds = xr.open_dataset(ds_path).astype(jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32)

    if debug:
        ds = ds.isel(shot=slice(0, 50))  # Limit to 50 shots

    # Ensure all required signals are present
    for signal in REQUIRED_SIGNALS:
        if signal not in ds:
            raise ValueError(f"Required signal for transport predictor training {signal} not found in dataset.")

    # Put dataset on a 50-point rho grid [0,1] for consistency (TCV original is ~200, C-Mod original is ~30)
    if "rho" not in ds.coords:
        raise ValueError("Dataset must have a 'rho' coordinate for interpolation.")
    rhogrid = np.linspace(0, 1, 50)
    ds = ds.interp(rho=rhogrid)

    # Calculate shape variables
    ds["ne_shape"] = ds["ne20_rho"] / ds["ne20_rho"].integrate("rho")
    ds["Te_shape"] = ds["Te_keV_rho"] / ds["Te_keV_rho"].integrate("rho")

    # Additional signals and duplicates for slight renames between submodules
    # This is for the individual submodule training to work, since when they're running on their own they expect these names.
    for signal in INPUT_POWER_SIGNALS:
        if signal not in ds:
            ds[signal] = xr.zeros_like(ds["Ip_MA"])

    ds["P_aux_MW"] = ds["P_NBI_MW"] + ds["P_ECRH_MW"] + ds["P_ICRF_MW"] + ds["P_LH_MW"]
    ds["P_abs_MW"] = ds["P_oh_MW"] + ds["P_aux_MW"]
    # Profile predictor
    ds["Paux_MW"] = ds["P_aux_MW"]
    ds["Ip"] = ds["Ip_MA"]
    # Power balance
    ds["delta"] = (ds["delta_top"] + ds["delta_bottom"]) / 2
    ds["ne19_line_avg"] = ds["ne20_line_avg"] * 10
    ds["epsilon"] = ds["a_minor"] / ds["R0"]

    # TODO(ZanderKeith): Later when we have multiple density treatments this should be handled better
    ds["ne20"] = ds["ne20_line_avg"]
    ds["ne19"] = ds["ne19_line_avg"]

    # If dataset was from a zarr store, must promote the 'time' data var to a coordinate
    if "time" not in ds.coords:
        ds = ds.set_coords("time")

    # Dataset retains all signals, the dataloader will filter out the ones that are not needed.
    return ds, "shot"
