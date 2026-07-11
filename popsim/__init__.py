import os
from importlib.resources import files
from pathlib import Path

# This import statement registers xarray types with JAX.
# Note: gdm xarray_jax also globally sets xr.set_options(arithmetic_compat="override").
import xarray_jax  # noqa: F401
from jax import config as jax_config

from popsim.config import config
from popsim.field_labels import discrete_no_save_field, discrete_time_field, no_save_field
from popsim.module_base import TimeDepModule, TimeIndepModule

# Default to 64-bit float unless user specifies otherwise
if os.getenv("JAX_ENABLE_X64") is None:
    jax_config.update("jax_enable_x64", True)

# Persistent XLA compilation cache on the shared filesystem
# Expensive modules take minutes to compile, so caching compiled executables saves
# the re-JIT on every resume. Set JAX_COMPILATION_CACHE_DIR to override, JAX
# reads that env var natively. Safe to delete the cache dir at any time.
if os.getenv("JAX_COMPILATION_CACHE_DIR") is None:
    jax_config.update("jax_compilation_cache_dir", os.path.join(os.path.expanduser("~"), ".cache", "jax_comp_cache"))
    jax_config.update("jax_persistent_cache_min_compile_time_secs", 10.0)


PACKAGE_ROOT = files("popsim")
SUBMODULES_DIR = os.path.join(PACKAGE_ROOT, "../submodules")
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")
ATOMIC_DATA_PATH = Path(os.path.join(PACKAGE_ROOT, "../atomic_data/output"))

__all__ = ["TimeDepModule", "TimeIndepModule", "config", "discrete_no_save_field", "discrete_time_field", "no_save_field"]
