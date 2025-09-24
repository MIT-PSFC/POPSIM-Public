import os

import h5py
import loguru
import numpy as np
import xarray as xr

from popsim.data.tcv.data_generators import generate_defuse_h5_paths
from popsim.interp import InterpType, interp


def _interpolate_signal(da_orig: xr.DataArray, timebase_new: np.ndarray, time_dim: str, time_coord: str) -> xr.DataArray:
    """Interpolate original DataArray onto new timebase

    Args:
        da_orig (xr.DataArray): The original DataArray to interpolate.
        timebase_new (np.ndarray): The new timebase to interpolate onto.
        time_dim (str): Name of the time dimension.
        time_coord (str): Name of the time coordinate.

    Returns:
        xr.DataArray: The interpolated DataArray.
    """
    signal = da_orig.name
    data_orig = da_orig.values
    time_orig = da_orig.coords[f"t_{signal}"].values

    # Check for malformed signals with non-monotonic time data
    if np.all(np.diff(time_orig) > 0):
        # Case 1: Data is fine, go ahead with interpolation
        if signal in ["Te_rho", "Ne_rho"]:
            # Replace NaNs in the profile data with 0
            data_orig = np.nan_to_num(data_orig, nan=0.0)
        elif np.isnan(data_orig[-1]).any() and sum(np.isnan(data_orig)) == 1:
            # Handle edge case where only the last value is NaN (semi-frequent on certain signals)
            data_orig = data_orig[:-1]
            time_orig = time_orig[:-1]

        # Rectilinear interpolation
        if data_orig.ndim == 1:
            # 1D data, straightforward interpolation
            interpolator = interp(time_orig, data_orig, interp_type=InterpType.RECTILINEAR)
            data_new = np.array(interpolator.evaluate(timebase_new))
        elif data_orig.ndim == 2:
            # 2D data, convert to PyTree format for vectorized interpolation and stack at end
            data_dict = {f"rho_{i}": data_orig[:, i] for i in range(data_orig.shape[1])}
            interpolator = interp(time_orig, data_dict, interp_type=InterpType.RECTILINEAR)
            data_dict_new = interpolator.evaluate(timebase_new)
            data_new = np.stack([np.array(data_dict_new[f"rho_{i}"]) for i in range(data_orig.shape[1])], axis=1)
        else:
            raise ValueError(f"Unsupported data dimensionality: {data_orig.ndim}")
    else:
        # Case 2: Malformed time coordinate, handle based on signal
        loguru.logger.debug(f"Non-monotonic time coordinate detected for {signal}")
        if signal in ["Te_rho", "Ne_rho"]:
            # Invalid time coord means invalid space coord too, can't fill with nans because we don't know what to put there
            raise ValueError(f"Malformed {signal}")
        else:
            loguru.logger.warning(f"Malformed time coordinate for {signal}, replacing with NaNs")
            data_new = np.full(len(timebase_new), np.nan, dtype=data_orig.dtype)

    if signal in ["Te_rho", "Ne_rho"]:
        rho = da_orig.coords[f"x_{signal}"].values
        da_new = xr.DataArray(data_new, dims=[time_dim, "rho"], coords={time_coord: (time_dim, timebase_new), "rho": rho})
    else:
        da_new = xr.DataArray(data_new, dims=[time_dim], coords={time_coord: (time_dim, timebase_new)})

    return da_new


def shot_from_path(h5_path: str) -> int:
    """Extract the shot number from the H5 file path."""
    # Example path: /path/to/defuse/shot_12345.h5
    filename = os.path.basename(h5_path)
    shot_str = filename.split("no")[1]
    return int(shot_str.split(".")[0])


def path_from_shot(shot: int) -> str:
    """Get the DEFUSE H5 file path for a given shot number."""
    h5_paths = list(generate_defuse_h5_paths())
    shots = [shot_from_path(path) for path in h5_paths]
    shot_index = shots.index(shot)
    h5_path = h5_paths[shot_index]
    return h5_path


def h5_to_xarray(
    h5_path: str,
    dataset_signals: dict[str, str],
    timebase_signal: str = "I_P",
    dt: float = 0.001,
    dtype=np.float32,
    time_dim: str = "time_idx",
    time_coord: str = "time",
    episode_dim: str = "shot",
) -> xr.Dataset:
    """Convert DEFUSE h5 file to xarray Dataset with requested signals. Uses rectilinear interpolation to put every signal on a uniform timebase

    Args:
        h5_path (str): Path to the DEFUSE h5 file.
        dataset_signals (dict): Dictionary mapping signal names to their corresponding paths in the h5 file.
        timebase_signal (str, optional): Name of the signal that will be used to determine the duration of the timebase
        dt (float, optional): Uniform dt for the timebase.
        dtype (np.dtype, optional): Data type for the output Dataset.
        time_dim (str, optional): Name of the time dimension in the output Dataset.
        time_coord (str, optional): Name of the time coordinate in the output Dataset.
        episode_dim (str, optional): Name of the episode dimension in the output Dataset.

    Raises:
        FileNotFoundError: If the h5 file is not found.
        ValueError: If one or more of the requested signals are not found in the dataset.

    Returns:
        xr.Dataset: Xarray Dataset containing the converted data.
    """

    shot = shot_from_path(h5_path)
    loguru.logger.info(f"Processing shot {shot} from {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        # If the /SIG group is not present, can't proceed
        if "/SIG" not in h5_file:
            raise ValueError(f"No /SIG group found in {h5_path}")
        sig_group = h5_file["/SIG"]

        ds_vars = {}
        for sig_name, defuse_name in dataset_signals.items():
            if defuse_name not in sig_group:
                raise ValueError(f"Signal {defuse_name} not found in {h5_path}")

            var_group = sig_group[defuse_name]
            time_coord_name = f"t_{sig_name}"

            if isinstance(var_group["signal"], h5py.Dataset):
                # Case 1: time + signal (1D signals)
                time_data = np.squeeze(var_group["time"][:])
                signal_data = np.squeeze(var_group["signal"][:])

                ds_vars[sig_name] = xr.DataArray(
                    signal_data.astype(dtype),
                    dims=[time_coord_name],
                    coords={time_coord_name: time_data},
                )
            elif isinstance(var_group["signal"], h5py.Group):
                # Case 2: signal group with t, x, z subfields (2D profiles)
                sig_subgroup = var_group["signal"]
                x_coord_name = f"x_{sig_name}"

                t_data = np.squeeze(sig_subgroup["t"][:])  # Time
                x_data = np.squeeze(sig_subgroup["x"][:])  # Spatial coordinate
                z_data = np.squeeze(sig_subgroup["z"][:])  # Data

                ds_vars[sig_name] = xr.DataArray(
                    z_data.astype(dtype),
                    dims=[time_coord_name, x_coord_name],
                    coords={
                        time_coord_name: t_data,
                        x_coord_name: x_data,
                    },
                )
            else:
                raise ValueError(f"Unexpected structure for signal {defuse_name} in {h5_path}")

        # Use the specified signal to define the time grid
        # Using linspace for precise uniform timebase to avoid floating-point accumulation errors
        ds_raw = xr.Dataset(ds_vars)
        max_time = ds_raw[f"t_{timebase_signal}"].values.max()
        n_points = int(max_time / dt) + 1  # +1 to include endpoint
        grid_max = np.floor(max_time / dt) * dt
        target_timebase = np.linspace(0, grid_max, n_points)
        if not np.isclose(np.diff(target_timebase).mean(), dt):
            raise ValueError(f"Target timebase is not uniform: {target_timebase}")

    # Make a new empty dataset with the right coords,
    # then fill it in with interpolated data for each signal
    ds_new = xr.Dataset(
        coords={
            time_coord: (time_dim, target_timebase),
            episode_dim: shot,
        },
    )
    for signal in ds_raw.data_vars:
        ds_new[signal] = _interpolate_signal(ds_raw[signal], target_timebase, time_dim, time_coord)

    return ds_new
