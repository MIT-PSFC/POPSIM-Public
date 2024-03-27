import os

import numpy as np
import xarray as xr

import popsim


def load_prd_transp_profiles() -> xr.Dataset:
    """Load the TRANSP 1D profiles, from the SPARC Public Reference Discharge.

    Returns:
        xr.Dataset: the TRANSP data contained in an xarray.
    """
    PATH_TO_TRANSP = os.path.join(popsim.SUBMODULES_DIR, "SPARCPublic", "PrimaryReferenceDischarge", "5 - transp_20221013.txt")
    data = {}
    with open(PATH_TO_TRANSP) as file:
        lines = file.readlines()
        current_variable = None
        for line in lines:
            line = line.strip()  # noqa: PLW2901
            if line.startswith("#"):  # New variable section
                current_variable = line.split("|")[0].strip("# ").strip()
                data[current_variable] = []
            elif line and current_variable:  # Data line
                _, value = line.split()
                data[current_variable].append(float(value))

    data_vars = {k: (["rho"], np.array(v)) for k, v in data.items() if k != "rho"}

    ds = xr.Dataset(data_vars, coords={"rho": data["rho"]})

    # Rename variables for units.
    ds = ds.rename({"te": "Te_keV", "ti": "Ti_keV", "ne": "ne19"})

    ds["ne20"] = 0.1 * ds["ne19"]
    return ds
