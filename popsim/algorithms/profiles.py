import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from cfspopcon.jax_compatible import density_peaking, plasma_profiles


class ProfileCalculator(eqx.Module):
    rho: ArrayLike

    def __init__(self, n_points: int):
        self.rho = jnp.linspace(0.0, 1.0, n_points)

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
        electron_temp_profile = jnp.maximum(electron_temp_profile, 0.01)
        ion_temp_profile = jnp.maximum(ion_temp_profile, 0.01)

        outs = {
            "electron_density_profile": electron_density_profile,
            "ion_density_profile": ion_density_profile,
            "electron_temp_profile": electron_temp_profile,
            "ion_temp_profile": ion_temp_profile,
            "effective_collisionality": effective_collisionality,
            "ion_density_peaking": ion_density_peaking,
            "electron_density_peaking": electron_density_peaking,
        }
        return outs
