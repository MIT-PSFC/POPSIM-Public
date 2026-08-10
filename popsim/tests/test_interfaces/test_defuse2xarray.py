import os
import tempfile

import numpy as np
import pytest
import xarray as xr

from popsim.data._paths import get_path_to_ml_data_dump
from popsim.interfaces.defuse2xarray import h5_to_xarray, path_from_shot

# Skip all tests in this module if ML data dump path is not available
pytestmark = pytest.mark.skipif(
    str(get_path_to_ml_data_dump()) != "/usr/local/mfe/ml_data_dump",
    reason="TCV data not available on this system",
    allow_module_level=True
)

DATASET_NAMES = [
    # Profiles
    "Te_rho",  # Electron temperature profile [eV]
    "Ne_rho",  # Electron density profile [m^-3]
    # Global quantities
    "I_P",      # Plasma current
    "BZERO",    # On-axis magnetic field
    "Wtot",     # Total stored energy (includes both thermal and fast ion energy, used in the calculation of BETAP)
    "NEavg",    # Line average electron density [m^-3]
    "a_minor",  # Plasma minor radius
    # Other
    "NBI",      # Neutral beam injection power
    "gas_valve",# Gas valve voltage control
]
DATASET_SIGNALS = {name: name for name in DATASET_NAMES}
DATASET_SIGNALS["gas_valve"] = "S_GAS/valve1/actual_ampl"

@pytest.mark.parametrize(
    "shot, requested_dataset_signals, should_pass", [
        (31650, DATASET_SIGNALS, False),
        (61237, DATASET_SIGNALS, True),
    ]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_h5_to_xarray(shot: int, requested_dataset_signals: dict, should_pass: bool, dtype: np.dtype):
    # Ensure the interface works on shots where all requested data is present
    # and properly errors out if the shot has missing data
    # TCV shots included in this test were manually curated

    h5_path = path_from_shot(shot)

    if should_pass:
        dataset = h5_to_xarray(h5_path, requested_dataset_signals, dtype=dtype)
        assert dataset is not None
        for signal in requested_dataset_signals.keys():
            assert signal in dataset
        # Ensure *everything* in the dataset is either int or the requested dtype
        for var in dataset.data_vars.values():
            assert var.dtype in [np.int32, dtype]
    else:
        with pytest.raises(ValueError):
            h5_to_xarray(h5_path, requested_dataset_signals, dtype=dtype)

