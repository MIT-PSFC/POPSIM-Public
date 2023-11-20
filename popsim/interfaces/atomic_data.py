"""Reads atomic data using cfspopcon but with Jax compatible interpolators."""
from functools import wraps
from typing import Union

import interpax
import jax.numpy as jnp
import xarray as xr

from cfspopcon import atomic_data
from popsim.interfaces.enums import Impurity


def atleast1d_inputs(func):
    """Decorator to ensure that all inputs to the function are at least 1D arrays."""

    @wraps(func)
    def wrapper(*args):
        # Convert all arguments to at least 1D
        new_args = [jnp.atleast_1d(arg) for arg in args]
        return func(*new_args)

    return wrapper


def _build_interpolator(curve: xr.Dataset) -> Union[interpax.Interpolator2D, interpax.Interpolator3D]:
    # By default, electron temperature is in eV and density is in 1e19.
    # "log" means log10.
    log_temp = jnp.array(curve.dim_log_electron_temperature)
    log_density = jnp.array(curve.dim_log_electron_density)

    if curve.ndim == 2:
        f = jnp.log10(jnp.array(curve.transpose("dim_log_electron_temperature", "dim_log_electron_density")))
        interp = interpax.Interpolator2D(
            x=log_temp,
            y=log_density,
            f=f,
            method="linear",
        )
    elif curve.ndim == 3:
        log_ne_tau = jnp.array(curve.dim_log_ne_tau)
        f = jnp.log10(
            jnp.array(
                curve.transpose(
                    "dim_log_electron_temperature",
                    "dim_log_electron_density",
                    "dim_log_ne_tau",
                )
            )
        )
        interp = interpax.Interpolator3D(
            x=log_temp,
            y=log_density,
            z=log_ne_tau,
            f=f,
            method="linear",
        )
    else:
        raise NotImplementedError(f"Cannot build an interpolator for a curve with ndim={curve.ndim}")
    return atleast1d_inputs(interp)


def read_atomic_data():
    data = atomic_data.read_atomic_data(build_interpolator=_build_interpolator)
    # Convert the enums to popsim types.
    data = {Impurity(k.value): v for k, v in data.items()}
    return data
