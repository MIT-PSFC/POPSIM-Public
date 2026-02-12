import glob
import os
import re
import time

import fire

from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.interfaces.meq2xarray.meq2xarray import tcv_db_to_xr, tcv_fbt_to_xr
from popsim.modules.meqml.fast_obs.config import FAST_OBS_CONFIG
from popsim.modules.meqml.fbt.config import FBT_SURROGATE_CONFIG
from popsim.modules.meqml.fgs.config import FGS_CONFIG
from popsim.modules.meqml.liuqe.config import LIUQE_CONFIG


def _parse_duration(s):
    """Parse a duration string like '1d', '12h', '30m', '2d12h' into seconds."""
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    total = 0
    for value, unit in re.findall(r"(\d+)([dhms])", s):
        total += int(value) * units[unit]
    if total == 0:
        raise ValueError(f"Invalid duration string: {s!r}. Use e.g. '1d', '12h', '30m', '2d12h'.")
    return total


def _resolve_paths(glob_str: str, recent: str | None) -> list[str]:
    """Glob for files and optionally filter to recently modified ones."""
    paths = glob.glob(glob_str)
    if recent is not None:
        cutoff = time.time() - _parse_duration(recent)
        paths = [p for p in paths if os.path.getmtime(p) >= cutoff]
    return paths


def build_fbt_ds(glob_str: str, zarr_path: str, extend_existing: bool = False, recent: str | None = None):
    """Build a tensorized dataset from FBT equilibrium files.

    Args:
        glob_str: Glob pattern matching input files, e.g. '/usr/local/mfe/ml_data_dump/TCV/meq_data/fbte/*'.
        zarr_path: Output path for the zarr dataset.
        extend_existing: If True, append to an existing zarr dataset instead of overwriting.
        recent: Only include files modified within this duration. Supports combinations of
            d (days), h (hours), m (minutes), s (seconds). E.g. '1d', '12h', '2d6h', '30m'.
    """
    paths = _resolve_paths(glob_str, recent)

    def process_fn(f):
        ds = tcv_fbt_to_xr(f)
        config = FBT_SURROGATE_CONFIG
        all_vars = (
            config["dataloader_config"]["input_vars"]
            + config["dataloader_config"]["target_vars"]
            + config["dataloader_config"]["extra_vars"]
        )
        all_vars = [v for v in all_vars if v in ds.data_vars]
        ds = ds[all_vars]
        return ds

    build_tensorized_dataset(
        process_fn=process_fn,
        identifiers=paths,
        zarr_path=zarr_path,
        time_dim="time_idx",
        episode_dim="shot",
        extend_existing=extend_existing,
        mb_per_chunk=50,
    )


def build_liuqe_ds(glob_str: str, zarr_path: str, extend_existing: bool = False, recent: str | None = None):
    """Build a tensorized dataset from LIUQE equilibrium files.

    Args:
        glob_str: Glob pattern matching input files, e.g. '/usr/local/mfe/ml_data_dump/TCV/meq_data/liuqe/*'.
        zarr_path: Output path for the zarr dataset.
        extend_existing: If True, append to an existing zarr dataset instead of overwriting.
        recent: Only include files modified within this duration. Supports combinations of
            d (days), h (hours), m (minutes), s (seconds). E.g. '1d', '12h', '2d6h', '30m'.
    """
    paths = _resolve_paths(glob_str, recent)

    def process_fn(f):
        dt = tcv_db_to_xr(f)
        ds = dt["liuqe"].to_dataset().isel(pq=-1).isel(pQ=-1)
        all_vars = set(
            LIUQE_CONFIG["dataloader_config"]["input_vars"]
            + LIUQE_CONFIG["dataloader_config"]["target_vars"]
            + LIUQE_CONFIG["dataloader_config"]["extra_vars"]
            + FGS_CONFIG["dataloader_config"]["input_vars"]
            + FGS_CONFIG["dataloader_config"]["target_vars"]
            + FGS_CONFIG["dataloader_config"]["extra_vars"]
            + FAST_OBS_CONFIG["dataloader_config"]["input_vars"]
            + FAST_OBS_CONFIG["dataloader_config"]["target_vars"]
            + FAST_OBS_CONFIG["dataloader_config"]["extra_vars"]
        )
        ds = ds[all_vars]
        return ds

    build_tensorized_dataset(
        process_fn=process_fn,
        identifiers=paths,
        zarr_path=zarr_path,
        time_dim="time_idx",
        episode_dim="shot",
        extend_existing=extend_existing,
        mb_per_chunk=50,
    )


if __name__ == "__main__":
    fire.Fire()
