import os

from jax import config

from popsim.module_base import ModuleBase

config.update("jax_enable_x64", True)

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
SUBMODULES_DIR = os.path.join(PACKAGE_ROOT, "../submodules")
TORAX_QLKNN_MODEL_PATH = os.path.join(SUBMODULES_DIR, "qlknn-hyper")

__all__ = ["ModuleBase"]
