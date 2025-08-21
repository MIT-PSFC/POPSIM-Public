import os
from collections.abc import Generator

import xarray as xr

from popsim.data._paths import get_path_to_ml_data_dump


def generate_defuse_h5_paths() -> Generator[str]:
    """Generator that yields paths to DEFUSE HDF5 files."""
    path_to_h5s = os.path.join(get_path_to_ml_data_dump(), "TCV/DEFUSE/DEFUSE_DB/DB_mat/TCV")

    if not os.path.exists(path_to_h5s):
        return

    all_files = os.listdir(path_to_h5s)
    for filename in all_files:
        if filename.endswith(".h5"):
            yield os.path.join(path_to_h5s, filename)


def generate_fbte_nc_paths() -> Generator[str]:
    """Generator that yields paths to FBTE netCDF files."""
    path_to_ncs = os.path.join(get_path_to_ml_data_dump(), "TCV/meq_data/fbte")

    if not os.path.exists(path_to_ncs):
        return

    all_files = os.listdir(path_to_ncs)
    for filename in all_files:
        if filename.endswith(".nc"):
            yield os.path.join(path_to_ncs, filename)


def generate_fbte_mat_paths() -> Generator[str]:
    """Generator that yields paths to FBTE MAT files."""
    path_to_mats = os.path.join(get_path_to_ml_data_dump(), "TCV/meq_data/fbte")

    if not os.path.exists(path_to_mats):
        return

    all_files = os.listdir(path_to_mats)
    for filename in all_files:
        if filename.endswith(".mat"):
            yield os.path.join(path_to_mats, filename)


def generate_defuse_h5s() -> Generator[dict[str, xr.Dataset]]:
    """Generator that yields dictionaries of xarray datasets for each DEFUSE HDF5 file.

    Yields:
        dict[str, xr.Dataset]: Each dictionary contains xarray datasets loaded from DEFUSE HDF5 files.
    """
    for h5_path in generate_defuse_h5_paths():
        defuse_dict = xr.open_groups(
            h5_path,
            engine="h5netcdf",
            phony_dims="sort",
        )
        yield defuse_dict


def generate_fbte_datasets() -> Generator[xr.Dataset]:
    """Generator that yields xarray datasets for each FBTE netCDF file.

    Yields:
        xr.Dataset: Each FBTE dataset loaded from netCDF files.
    """
    for nc_path in generate_fbte_nc_paths():
        yield xr.open_dataset(nc_path)


def generate_fbte_mat_datasets() -> Generator[xr.Dataset]:
    pass
