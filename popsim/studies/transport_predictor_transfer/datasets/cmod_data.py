import os

import loguru
import xarray as xr

from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.studies.transport_predictor_transfer.datasets.workflow import (
    EPISODE_DIM,
    TIME_COORD,
    TIME_DIM,
    DataWorkflow,
)

SUMMARY_TABLE = "summary"
IPMAX = 100e3  # [A]
PULSE_LENGTH = 0.1  # [s]
MIN_SHOT = 1050204013
MAX_SHOT = 1160930043

CMOD_DATASET_SIGNALS = [
    # Profiles being predicted
    "te_rho",  # Electron temperature profile [eV]
    "ne_rho",  # Electron density profile [m^-3]
    # Global quantities
    "ip",  # Plasma current
    "btor",  # On-axis magnetic field
    "wmhd",  # Total stored energy (TODO(ZanderKeith): I don't think C-Mod has a consistent fast particle measurement, so this is all we've got)
    "beta_p",  # Plasma beta
    "n_e",  # Line average electron density [m^-3]
    "a_minor",  # Plasma minor radius
    "kappa",  # Plasma elongation
    "tritop",  # Top triangularity
    "tribot",  # Bottom triangularity
    "rmagx",  # Major radius [m]
    # Power sources and sinks
    "p_oh",  # Ohmic heating power
    "p_rad",  # Bulk radiated heating power
    "p_icrf",  # ICRF heating power
    "p_lh",  # Lower hybrid heating power (yes this is actually lower hybrid on C-Mod, NOT the LH transition threshold like on TCV)
    # Other
    # TODO(ZanderKeith): Add gas valves when we get to that point
]

CMOD_SIGNAL_MAP = {
    # Simple renames
    "R0": ("rmagx", None, "rename"),
    "delta_top": ("tritop", None, "rename"),
    "delta_bottom": ("tribot", None, "rename"),
    # Conversions
    "B0": ("btor", None, "abs"),
    "Ip_MA": ("ip", 1e-6, "abs_multiply"),
    "P_oh_MW": ("p_oh", 1e-6, "multiply"),
    "P_rad_MW": ("p_rad", 1e-6, "multiply"),
    "ne20_line_avg": ("n_e", 1e-20, "multiply"),
    "Wtot_MJ": ("wmhd", 1e-6, "multiply"),
    "ne20_rho": ("ne_rho", 1e-20, "multiply"),
    "Te_keV_rho": ("te_rho", 1e-3, "multiply"),
    "P_ICRF_MW": ("p_icrf", 1e-3, "multiply"),
    "P_LH_MW": ("p_lh", 1e-3, "multiply"),
}

CMOD_SIGNAL_BOUNDS = {
    "Wtot_MJ": {"bounds": (1e-3, None)},
    "beta_p": {"bounds": (0, 1)},
    "P_rad_MW": {"bounds": (0, 2.0)},
}


class CMODDataWorkflow(DataWorkflow):
    signal_map = CMOD_SIGNAL_MAP
    signal_bounds = CMOD_SIGNAL_BOUNDS

    def build_dataset(self, extend_existing: bool = False):
        """Use Disruption-Py to build a raw dataset"""
        # Only import disruption-py if we are actually using it, since MDSPlus is only available on data clusters
        from disruption_py.machine.tokamak import Tokamak
        from disruption_py.settings import RetrievalSettings
        from disruption_py.workflow import get_shots_data

        from popsim.studies.transport_predictor_transfer.datasets.dispy_utils import get_shotlist_from_sql

        build_handler = loguru.logger.add(os.path.join(self.raw_ds_dir, f"build_{self.ds_name}.log"))
        loguru.logger.info(f"Building CMOD dataset {self.ds_name} in {self.raw_ds_dir}")

        retrieval_settings = RetrievalSettings(
            run_columns=CMOD_DATASET_SIGNALS,
            time_setting="tmdb",
            only_requested_columns=True,
        )

        def _process_fn(shot_id: int) -> xr.Dataset:
            dispy_ds = get_shots_data(
                tokamak=Tokamak.CMOD,
                shotlist_setting=[shot_id],
                retrieval_settings=retrieval_settings,
                output_setting="dataset",
                num_processes=1,
            )
            shot_ds = dispy_ds.where(dispy_ds["shot"] == shot_id, drop=True)
            if shot_ds["ne_rho"].isnull().all() or shot_ds["te_rho"].isnull().all():
                # The whole point is to predict these, so skip shots that don't have them
                loguru.logger.warning(f"Shot {shot_id} has no ne_rho or te_rho data, skipping")
                return None
            # Reformat the dimensions and coords to match POPSIM conventions
            shot_ds = shot_ds.rename(
                {
                    "idx": TIME_DIM,
                    "time": TIME_COORD,
                }
            )
            shot_ds = shot_ds.drop_vars("shot")
            shot_ds = shot_ds.expand_dims({EPISODE_DIM: [shot_id]})
            return shot_ds

        shotlist = get_shotlist_from_sql(
            summary_table=SUMMARY_TABLE,
            ipmax=IPMAX,
            pulse_length=PULSE_LENGTH,
            min_shot=self.min_shot_id,
            max_shot=MAX_SHOT,
            num_shots=self.max_num_shots,
        )
        loguru.logger.info(f"Found {len(shotlist)} candidate shots")

        dataset = build_tensorized_dataset(
            process_fn=_process_fn,
            identifiers=shotlist,
            zarr_path=self.raw_ds_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=extend_existing,
            mb_per_chunk=None,
        )

        loguru.logger.info(f"Completed building raw dataset\n{dataset}")
        try:
            loguru.logger.remove(build_handler)
        except ValueError:
            pass
