import jax.numpy as jnp

import popsim.algorithms.density as density_model
from cfspopcon.jax_compatible import impurity_effects, radiated_power
from popsim.enums import AtomicNumberMap, FuelSpecies, Impurity, Species
from popsim.interfaces.atomic_data import RadasCurves, RadasCurvesForSpecies
from popsim.tree_util import leaves_as_array


def calc_impurity_state(
    density_state: density_model.State, average_pressure_kev_1e19: float, radas_curves: RadasCurves
) -> dict[str, float]:
    """Calculate the effective charge, dilution, and volume-averaged electron density given particle densities and the average electron pressure.

    Args:
        density_state (density_model.State): state of the density_model.
        average_pressure_kev_1e19 (float): average pressure in keV-1e19.
        radas_curves (RadasCurves): dictionary containing the radas curves for each impurity species.

    Returns:
        dict[str, float]: dictionary containing the effective charge, dilution, and volume-averaged electron density.
    """
    # Calculate the volume-averaged electron density assuming full ionization.
    # We then use this "guess" electron density to calculate the charge state.
    # The unit test "test_sensitivity_to_electron_density" shows that the charge state
    # is highly insensitive to the guess electron density.
    electron_density_19_full_ion_dict = {
        species: float(AtomicNumberMap[species]) * species_density for species, species_density in density_state.vol_avg_ion.items()
    }
    average_electron_density_19_guess = jnp.sum(leaves_as_array(electron_density_19_full_ion_dict))
    average_electron_temp_kev_guess = (0.5 * average_pressure_kev_1e19) / average_electron_density_19_guess

    def calc_charge_state(species: Species) -> float:
        if isinstance(species, Impurity):
            return calc_impurity_charge_state(
                average_electron_density_19_guess,
                average_electron_temp_kev_guess,
                radas_curves[species],
            )
        elif isinstance(species, FuelSpecies):
            return 1.0
        else:
            raise ValueError(f"Species {species} is not a valid species.")

    charge_states = {species: calc_charge_state(species) for species in density_state.vol_avg_ion.keys()}

    electron_density_19 = jnp.sum(
        jnp.array([charge_state * density_state.vol_avg_ion[species] for species, charge_state in charge_states.items()])
    )

    # Calculate the dilution.
    dilution = density_state.total_volume_average_ion_density / electron_density_19

    # Compute the z-effective terms for each species.
    zeff_terms = {
        species: zeff_term(AtomicNumberMap[species], density_state.vol_avg_ion[species], electron_density_19)
        for species in density_state.vol_avg_ion.keys()
    }

    species_concentrations = {
        species: density_state.vol_avg_ion[species] / electron_density_19 for species in density_state.vol_avg_ion.keys()
    }

    outs = {
        "z_effective": jnp.sum(leaves_as_array(zeff_terms)),
        "dilution": dilution,
        "volume_average_electron_density_19": electron_density_19,
        "species_concentrations": species_concentrations,
    }
    return outs


def calc_impurity_radiated_power_radas(
    electron_temp_profile,
    electron_density_profile,
    impurity_concentrations,
    volume_integrator,
    radas_curves: RadasCurves,
):
    def species_calc(species, concentration):
        return radiated_power.calc_impurity_radiated_power_radas(
            electron_temp_profile=electron_temp_profile,
            electron_density_profile=electron_density_profile,
            impurity_concentration=concentration,
            volume_integrator=volume_integrator,
            Lz_curve=radas_curves[species].coronal_Lz_interpolator,
        )

    species_components = {species: species_calc(species, concentration) for species, concentration in impurity_concentrations.items()}

    total_power = jnp.sum(leaves_as_array(species_components))
    return total_power, species_components


def zeff_term(charge_state: float, species_density: float, electron_density: float) -> float:
    return (charge_state**2 * species_density) / electron_density


def calc_impurity_charge_state(
    average_electron_density_19: float, average_electron_temp_keV: float, radas_curves: RadasCurvesForSpecies
) -> float:
    """
    Wrapper for impurity_effects.calc_impurity_charge_state_impl.
    """
    return impurity_effects.calc_impurity_charge_state_impl(
        1e19 * average_electron_density_19, 1000.0 * average_electron_temp_keV, radas_curves.coronal_mean_Z_interpolator
    )
