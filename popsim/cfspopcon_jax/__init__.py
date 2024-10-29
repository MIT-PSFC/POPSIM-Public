from jax import config

config.update("jax_enable_x64", True)

__all__ = [
    "beta",
    "geometry",
    "impurity_effects",
    "plasma_profiles",
]
