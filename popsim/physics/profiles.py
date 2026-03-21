from functools import wraps

import chex
import interpax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
from jaxtyping import Array
from numpy import float64
from numpy.typing import NDArray

from popsim.cfspopcon_jax import density_peaking, plasma_profiles
from popsim.cfspopcon_jax.plasma_profile_data import density_and_temperature_profile_fits
from popsim.enums import ProfileForm


def _atleast1d_inputs(func):
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

    return _atleast1d_inputs(interp)


def _read_prf_profiles(dataset: str = "PRF"):
    width_interpolator, aLT_interpolator = density_and_temperature_profile_fits.read_prf_data(
        dataset=dataset, build_interpolator=_build_interpolator
    )
    return width_interpolator, aLT_interpolator


@chex.dataclass
class PRFProfiles:
    width_interpolator: interpax.Interpolator2D
    aLT_interpolator: interpax.Interpolator2D

    def __init__(self):
        """Get interpolators"""
        self.width_interpolator, self.aLT_interpolator = _read_prf_profiles()

    def __call__(
        self,
        average_electron_density: float,
        average_electron_temp: float,
        average_ion_temp: float,
        electron_density_peaking: float,
        ion_density_peaking: float,
        temperature_peaking: float,
        dilution: float,
        rho: NDArray[float64],
        normalized_inverse_temp_scale_length: float,
    ):
        rho, electron_temp_profile, electron_density_profile = self.evaluate_density_and_temperature_profile_fits(
            T_avol=average_electron_temp,
            n_avol=average_electron_density,
            temperature_peaking=temperature_peaking,
            nu_n=electron_density_peaking,
            aLT=normalized_inverse_temp_scale_length,
            rho=rho,
        )
        rho, ion_temp_profile, ion_density_profile = self.evaluate_density_and_temperature_profile_fits(
            T_avol=average_ion_temp,
            n_avol=average_electron_density * dilution,
            temperature_peaking=temperature_peaking,
            nu_n=ion_density_peaking,
            aLT=normalized_inverse_temp_scale_length,
            rho=rho,
        )

        return electron_temp_profile, electron_density_profile, ion_temp_profile, ion_density_profile

    def evaluate_density_and_temperature_profile_fits(
        self,
        T_avol: float,
        n_avol: float,
        temperature_peaking: float,
        nu_n: float,
        aLT: float = 2.0,
        width_ped: float = 0.05,
        rho: NDArray[float64] | None = None,
    ) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:  # TODO: fill out docstring
        """Evaluate temperature-density profile fits."""

        # ---- Find parameters consistent with peaking
        x_a = self.width_interpolator(aLT, temperature_peaking)[0]
        aLn = self.aLT_interpolator(x_a, nu_n)[0]

        # ---- Evaluate profiles
        x, T, _ = density_and_temperature_profile_fits.evaluate_profile(T_avol, width_ped=width_ped, aLT_core=aLT, width_axis=x_a, rho=rho)
        x, n, _ = density_and_temperature_profile_fits.evaluate_profile(n_avol, width_ped=width_ped, aLT_core=aLn, width_axis=x_a, rho=rho)

        return x, T, n


@chex.dataclass
class ProfileCalculator:
    density_profile_form: ProfileForm
    temp_profile_form: ProfileForm
    rho: Array
    PRFcalc: PRFProfiles

    def __init__(
        self,
        density_profile_form: ProfileForm,
        temp_profile_form: ProfileForm,
        rho: Array,
        PRFcalc=PRFProfiles,
    ):
        self.density_profile_form = density_profile_form
        self.temp_profile_form = temp_profile_form
        self.rho = rho
        self.PRFcalc = PRFcalc()

    def __call__(
        self,
        average_electron_density_19: float,
        average_electron_temp_keV: float,
        average_ion_temp_keV: float,
        ion_density_peaking_offset: float,
        electron_density_peaking_offset: float,
        temperature_peaking: float,
        major_radius: float,
        z_effective: float,
        dilution: float,
        beta_toroidal: float,
        normalized_inverse_temp_scale_length: float,
    ):
        effective_collisionality = density_peaking.calc_effective_collisionality(
            average_electron_density_19, average_electron_temp_keV, major_radius, z_effective
        )

        ion_density_peaking = density_peaking.calc_density_peaking(
            effective_collisionality, beta_toroidal, nu_noffset=ion_density_peaking_offset
        )

        electron_density_peaking = density_peaking.calc_density_peaking(
            effective_collisionality, beta_toroidal, nu_noffset=electron_density_peaking_offset
        )

        # density profiles
        if self.density_profile_form.value == ProfileForm.analytic.value:
            (
                _,
                electron_density_profile,
                ion_density_profile,
                electron_temp_profile,
                ion_temp_profile,
            ) = plasma_profiles.calc_analytic_profiles(
                average_electron_density_19,
                average_electron_temp_keV,
                average_ion_temp_keV,
                electron_density_peaking,
                ion_density_peaking,
                temperature_peaking,
                dilution,
                self.rho,
            )

            # Bit of a hack to avoid zero at the edge which seems to cause problems.
            electron_density_profile = jnp.maximum(electron_density_profile, 0.01)
            ion_density_profile = jnp.maximum(ion_density_profile, 0.01)

        elif self.density_profile_form.value == ProfileForm.prf.value:
            (_, electron_density_profile, _, ion_density_profile) = self.PRFcalc(
                average_electron_density_19,
                average_electron_temp_keV,
                average_ion_temp_keV,
                electron_density_peaking,
                ion_density_peaking,
                temperature_peaking,
                dilution,
                self.rho,
                normalized_inverse_temp_scale_length,
            )

        # temperature profiles
        if self.temp_profile_form.value == ProfileForm.analytic.value:
            (
                _,
                electron_density_profile,
                ion_density_profile,
                electron_temp_profile,
                ion_temp_profile,
            ) = plasma_profiles.calc_analytic_profiles(
                average_electron_density_19,
                average_electron_temp_keV,
                average_ion_temp_keV,
                electron_density_peaking,
                ion_density_peaking,
                temperature_peaking,
                dilution,
                self.rho,
            )

            # Bit of a hack to avoid zero at the edge which seems to cause problems.
            electron_temp_profile = jnp.maximum(electron_temp_profile, 0.01)
            ion_temp_profile = jnp.maximum(ion_temp_profile, 0.01)

        elif self.temp_profile_form.value == ProfileForm.prf.value:
            (electron_temp_profile, _, ion_temp_profile, _) = self.PRFcalc(
                average_electron_density_19,
                average_electron_temp_keV,
                average_ion_temp_keV,
                electron_density_peaking,
                ion_density_peaking,
                temperature_peaking,
                dilution,
                self.rho,
                normalized_inverse_temp_scale_length,
            )

        else:
            raise NotImplementedError(f"Profile form {self.profile_form} is not recognized...")

        outs = {
            "rho": xr.Variable(dims=("rho",), data=self.rho),
            "electron_density_profile": xr.DataArray(electron_density_profile, dims=["rho"]),
            "ion_density_profile": xr.DataArray(ion_density_profile, dims=["rho"]),
            "electron_temp_profile": xr.DataArray(electron_temp_profile, dims=["rho"]),
            "ion_temp_profile": xr.DataArray(ion_temp_profile, dims=["rho"]),
            "effective_collisionality": effective_collisionality,
            "ion_density_peaking": ion_density_peaking,
            "electron_density_peaking": electron_density_peaking,
        }

        return outs
