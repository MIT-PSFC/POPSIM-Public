import os

import loguru
import numpy as np
import xarray as xr

from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.data.tcv.data_generators import generate_defuse_h5_paths
from popsim.interfaces.defuse2xarray import h5_to_xarray, shot_from_path
from popsim.studies.transport_predictor_transfer.datasets.workflow import (
    DATA_TYPE,
    DATASET_DT,
    EPISODE_DIM,
    TIME_COORD,
    TIME_DIM,
    DataWorkflow,
)

TCV_DATASET_SIGNALS = [
    # Profiles being predicted
    "Te_rho",  # Electron temperature profile [eV]
    "Ne_rho",  # Electron density profile [m^-3]
    # Global quantities
    "I_P",  # Plasma current
    "BZERO",  # On-axis magnetic field
    "Wtot",  # Total stored energy (includes both thermal and fast ion energy, used in the calculation of BETAP)
    "BETAP",  # Plasma beta
    "NEavg",  # Line average electron density [m^-3]
    "a_minor",  # Plasma minor radius
    "KAPPA",  # Plasma elongation
    "DELTA_TOP",  # Top triangularity
    "DELTA_BOTTOM",  # Bottom triangularity
    "RMAG",  # Major radius [m]
    # Power sources and sinks
    "POHM",  # Ohmic heating power
    "PradBulk",  # Bulk radiated heating power
    "NBI",  # Neutral beam injection power
    "ECRH",  # Electron cyclotron resonance heating power
    # Other
    "gas_valve",  # Gas valve voltage control
    # Comparison
    "Ne_edge_avg",  # Line average electron density at the edge [m^-3]
    "TAU_conf",  # Confinement time [s]
    "TAU_conf_calc",  # Something mimicking real-time confinement time calculation [s]
    "P_LH",  # LH transition threshold
]

TCV_SIGNAL_BOUNDS = {
    "ne20_line_avg": {"bounds": (0.1, 2e20)},
    "Wtot_MJ": {"bounds": (1e-3, None)},
    "P_rad_MW": {"bounds": (0.001, 2.0)},
    "Te_keV_rho": {"bounds": (0.0, 0.25), "select": {"rho": 1.0}},
    "ne20_rho": {"bounds": (0.0, 0.4), "select": {"rho": 1.0}},
}


class TCVDataWorkflow(DataWorkflow):
    signal_bounds = TCV_SIGNAL_BOUNDS

    def build_dataset(self, extend_existing: bool = False):
        """Build the dataset from raw data."""

        build_handler = loguru.logger.add(os.path.join(self.raw_ds_dir, f"build_{self.ds_name}.log"))
        loguru.logger.info(f"Building TCV dataset {self.ds_name} in {self.raw_ds_dir}")

        all_h5_paths = list(generate_defuse_h5_paths())
        loguru.logger.info(f"Found {len(all_h5_paths)} HDF5 files.")

        # Filter h5 paths based on shot number
        all_shots = [shot_from_path(p) for p in all_h5_paths]
        selected_shots = sorted([shot for shot in all_shots if self.min_shot_id <= shot])
        if self.max_num_shots is not None:
            selected_shots = selected_shots[: self.max_num_shots]
        shot_indices = [i for i, shot in enumerate(all_shots) if shot in selected_shots]
        selected_h5_paths = [all_h5_paths[i] for i in shot_indices]
        loguru.logger.info(f"Selected the {len(selected_h5_paths)} shots after {self.min_shot_id}")

        # Going from the names in the dataset_signals to the name in the TCV dataset
        dataset_signals = {signal: signal for signal in TCV_DATASET_SIGNALS}
        dataset_signals["gas_valve"] = "S_GAS/valve1/actual_ampl"
        loguru.logger.info(f"Using the following signals:\n{dataset_signals}")

        def _process_fn(h5_path: str) -> xr.Dataset:
            return h5_to_xarray(
                h5_path,
                dataset_signals,
                timebase_signal="I_P",
                dt=DATASET_DT,
                dtype=DATA_TYPE,
                time_dim=TIME_DIM,
                time_coord=TIME_COORD,
                episode_dim=EPISODE_DIM,
            )

        dataset = build_tensorized_dataset(
            process_fn=_process_fn,
            identifiers=selected_h5_paths,
            zarr_path=self.raw_ds_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=extend_existing,
            mb_per_chunk=None,
        )

        loguru.logger.info(f"Completed building raw dataset\n{dataset}")
        loguru.logger.remove(build_handler)

    def process_signals(self, ds: xr.Dataset) -> xr.Dataset:
        # Simple renames
        ds = ds.rename(
            {
                "RMAG": "R0",
                "KAPPA": "kappa",
                "DELTA_TOP": "delta_top",
                "DELTA_BOTTOM": "delta_bottom",
                "P_LH": "LH_transition_threshold_MW",
            }
        )

        # Conversions
        ds["B0"] = np.abs(ds["BZERO"])
        ds["Ip_MA"] = np.abs(ds["I_P"]) * 1e-6
        ds["P_oh_MW"] = ds["POHM"] * 1e-6
        ds["P_rad_MW"] = ds["PradBulk"] * 1e-6
        ds["ne20_line_avg"] = ds["NEavg"] * 1e-20
        ds["Wtot_MJ"] = ds["Wtot"] * 1e-6
        ds["ne20_rho"] = ds["Ne_rho"] * 1e-20
        ds["Te_keV_rho"] = ds["Te_rho"] * 1e-3

        # If the signal is not present, create it as zeros
        ds["P_NBI_MW"] = ds["NBI"].fillna(0.0)
        ds["P_ECRH_MW"] = ds["ECRH"].fillna(0.0)

        # Drop old names
        ds = ds.drop_vars(["BZERO", "I_P", "POHM", "PradBulk", "NEavg", "Wtot", "Te_rho", "Ne_rho", "NBI", "ECRH"])

        return ds
