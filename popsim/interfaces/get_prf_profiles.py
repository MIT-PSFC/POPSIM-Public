"""Reads PRF profiles using cfspopcon but with Jax compatible interpolators."""
from functools import wraps

import interpax
import jax.numpy as jnp
import pandas as pd

from cfspopcon.jax_compatible.plasma_profile_data import density_and_temperature_profile_fits


def atleast1d_inputs(func):
    """Decorator to ensure that all inputs to the function are at least 1D arrays."""

    @wraps(func)
    def wrapper(*args):
        # Convert all arguments to at least 1D
        new_args = [jnp.atleast_1d(arg) for arg in args]
        return func(*new_args)

    return wrapper


def _build_interpolator(df: pd.DataFrame) -> interpax.Interpolator2D:
    # By default, electron temperature is in eV and density is in 1e19.
    # "log" means log10.
    x = jnp.array([jnp.float64(x[1]) for x in df.columns.values])
    y = jnp.array([jnp.float64(x[1]) for x in df.index.values])
    f = jnp.array(df.T.values)

    interp = interpax.Interpolator2D(
        x=x,
        y=y,
        f=f,
        method="linear",
        extrap=True,
    )

    return atleast1d_inputs(interp)


def read_prf_profiles(dataset: str = "PRF"):
    width_interpolator, aLT_interpolator = density_and_temperature_profile_fits.read_prf_data(
        dataset=dataset, build_interpolator=_build_interpolator
    )
    return width_interpolator, aLT_interpolator
