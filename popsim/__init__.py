import os
from importlib.resources import files

from jax import config

from popsim.hybrid_state import discrete_time_field
from popsim.module_base import ModuleBase

config.update("jax_enable_x64", True)


PACKAGE_ROOT = files("popsim")
SUBMODULES_DIR = os.path.join(PACKAGE_ROOT, "../submodules")
TORAX_QLKNN_MODEL_PATH = os.path.join(SUBMODULES_DIR, "qlknn-hyper")
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")

__all__ = ["ModuleBase", "discrete_time_field"]
