import os
from importlib.resources import files
from pathlib import Path

# This import statement registers xarray types with JAX
import xarray_jax  # noqa: F401
from jax import config as jax_config

from popsim.config import config
from popsim.field_labels import discrete_no_save_field, discrete_time_field, no_save_field
from popsim.module_base import TimeDepModule, TimeIndepModule

jax_config.update("jax_enable_x64", True)


PACKAGE_ROOT = files("popsim")
SUBMODULES_DIR = os.path.join(PACKAGE_ROOT, "../submodules")
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")
ATOMIC_DATA_PATH = Path(os.path.join(PACKAGE_ROOT, "../atomic_data/output"))

__all__ = ["TimeDepModule", "TimeIndepModule", "config", "discrete_no_save_field", "discrete_time_field", "no_save_field"]
