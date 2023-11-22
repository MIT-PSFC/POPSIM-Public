import os

from jax import config

config.update("jax_enable_x64", True)

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
