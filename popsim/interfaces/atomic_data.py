"""Reads atomic data using cfspopcon but with Jax compatible interpolators."""
import typing
from functools import wraps
from pathlib import Path
from typing import Union

import chex
import interpax
import jax.numpy as jnp
import xarray as xr
from cfspopcon import atomic_data

from popsim import ATOMIC_DATA_PATH
from popsim.enums import Impurity


@chex.dataclass
class RadasCurvesForSpecies:
    coronal_mean_Z_interpolator: typing.Callable[[float], float]
    coronal_Lz_interpolator: typing.Callable[[float], float]


RadasCurves = dict[Impurity, RadasCurvesForSpecies]


def atleast1d_inputs(func):
    """Decorator to ensure that all inputs to the function are at least 1D arrays."""

    @wraps(func)
    def wrapper(*args):
        # Convert all arguments to at least 1D
        new_args = [jnp.atleast_1d(arg) for arg in args]
        return func(*new_args)

    return wrapper


def _build_interpolator(curve: xr.Dataset) -> Union[interpax.Interpolator2D, interpax.Interpolator3D]:
    # Strip away pint units as they are currently not working with Jax.
    # https://github.com/cfs-energy-internal/POPSIM/issues/3
    curve = curve.pint.dequantify()

    # By default, electron temperature is in eV and density is in 1e19.
    # "log" means log10.
    log_temp = jnp.array(curve.dim_log_electron_temp)
    log_density = jnp.array(curve.dim_log_electron_density)

    if curve.ndim == 2:
        f = jnp.log10(jnp.array(curve.transpose("dim_log_electron_temp", "dim_log_electron_density")))
        interp = interpax.Interpolator2D(
            x=log_temp,
            y=log_density,
            f=f,
            method="linear",
            # TODO(allenw): RADAS data currently does not support
            # our full range, so we have to extrap.
            extrap=True,
        )
    elif curve.ndim == 3:
        log_ne_tau = jnp.array(curve.dim_log_ne_tau)
        f = jnp.log10(
            jnp.array(
                curve.transpose(
                    "dim_log_electron_temp",
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
            # TODO(allenw): RADAS data currently does not support
            # our full range, so we have to extrap.
            extrap=True,
        )
    else:
        raise NotImplementedError(f"Cannot build an interpolator for a curve with ndim={curve.ndim}")
    return atleast1d_inputs(interp)


def read_atomic_data() -> RadasCurves:
    atomic_data_path = Path(ATOMIC_DATA_PATH)

    # Check that the path exists.
    if atomic_data_path.exists() is False:
        raise FileNotFoundError(f"Could not find the atomic data directory {atomic_data_path}. Try running build_radas.sh.")

    data = atomic_data.read_atomic_data(Path(ATOMIC_DATA_PATH), build_interpolator=_build_interpolator)
    # Convert the enums to popsim types.
    data = {
        Impurity(k.value): RadasCurvesForSpecies(
            coronal_mean_Z_interpolator=v.coronal_mean_Z_interpolator,
            coronal_Lz_interpolator=v.coronal_Lz_interpolator,
        )
        for k, v in data.items()
    }
    return data
