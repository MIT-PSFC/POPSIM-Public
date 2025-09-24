import os
import warnings
from pathlib import Path

SUPPORTED_ML_DATA_PATHS = [
    Path("/usr/local/mfe/ml_data_dump/"),
    Path("/orcd/nese/psfc/001/allenw/ml_data_dump/"),
    Path("/fusion/projects/disruption_warning/data/popsim/"),
]

SUPPORTED_ML_SCRATCH_PATHS = [
    Path("/usr/local/mfe/ml_data_dump/"),
    Path(f"/home/{os.getlogin()}/orcd/scratch/"),
    Path(f"/cscratch/{os.getlogin()}/"),
]


def get_path_to_ml_data_dump() -> Path | None:
    """Get the path to the ML data dump on supported systems.

    Returns:
        Path | None: Path object pointing to the ML data dump directory,
                     or None if no path is found (with warning).

    Raises:
        ValueError: If multiple paths exist.
    """
    existing_paths = [path for path in SUPPORTED_ML_DATA_PATHS if path.exists()]

    if len(existing_paths) > 1:
        raise ValueError(f"Multiple ML data dump paths exist: {existing_paths}")
    elif len(existing_paths) == 1:
        return existing_paths[0]
    else:
        warnings.warn(f"No ML data dump path found. Checked: {SUPPORTED_ML_DATA_PATHS}", stacklevel=2)
        return None


def get_path_to_ml_data_scratch() -> Path | None:
    """Get the path to the ML data scratch space on supported systems.

    Returns:
        Path | None: Path object pointing to the ML data scratch directory,
                     or None if no path is found (with warning).

    Raises:
        ValueError: If multiple paths exist.
    """
    existing_paths = [path for path in SUPPORTED_ML_SCRATCH_PATHS if path.exists()]

    if len(existing_paths) > 1:
        raise ValueError(f"Multiple ML data scratch paths exist: {existing_paths}")
    elif len(existing_paths) == 1:
        return existing_paths[0]
    else:
        warnings.warn(f"No ML data scratch path found. Checked: {SUPPORTED_ML_SCRATCH_PATHS}", stacklevel=2)
        return None
