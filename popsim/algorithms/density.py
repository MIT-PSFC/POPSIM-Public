from collections.abc import Sequence
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from cfspopcon.jax_compatible import impurity_effects, radiated_power
from popsim.enums import AtomicNumberMap, FuelSpecies, Impurity, Species
from popsim.interfaces.atomic_data import read_atomic_data
from popsim.tree_util import leaves_as_array


class GenericDensityModel(eqx.Module):
    """
    A model for volume-averaged ion density evolution in a plasma.
    Implements an equation of the form:
        d/dt(n_i * V) = -n_i * V / tau + sum(sources) - sum(sinks)
    """

    species: Species

    class State(eqx.Module):
        volume_average_density: float  # 1e19/m^3

    class Params(eqx.Module):
        sources_and_sinks: PyTree[float]  # PyTree of net particle fluxes from various sources and sinks 1e19/s
        species_confinement_time: float  # seconds
        volume_dot: float  # m^3/s
        volume: float  # m^3

    def __init__(self, species: Species):
        self.species = species

    def __call__(self, state: State, params: Params) -> State:
        N = params.volume * state.volume_average_density

        sources_and_sinks = jnp.array(jax.tree_util.tree_leaves(params.sources_and_sinks))
        net_particle_flux = jnp.sum(sources_and_sinks)
        N_dot = -N / params.species_confinement_time + net_particle_flux

        # By the quotient rule:
        n_dot = (N_dot * params.volume - N * params.volume_dot) / params.volume**2
        state_dot = GenericDensityModel.State(volume_average_density=n_dot)

        debugs = {
            "N": N,
            "net_particle_flux": net_particle_flux,
            "N_dot": N_dot,
            "n_dot": n_dot,
        }
        return state_dot, debugs


class MultiSpeciesDensityModel(eqx.Module):
    """
    Wrapper class to handle multiple instances of the GenericDensityModel for different species.
    """

    species_models: dict[Species, GenericDensityModel]

    class State(eqx.Module):
        vol_avg_ion: dict[Species, float]  # 1e19/m^3

        @property
        def total_volume_average_ion_density(self) -> float:
            return jnp.sum(leaves_as_array(self.vol_avg_ion))

    class Params(eqx.Module):
        sources_and_sinks: dict[Species, PyTree[float]]  # PyTree of net particle fluxes from various sources and sinks 1e19/s
        species_confinement_time: dict[Species, float]  # species confinement time in seconds
        volume_dot: float  # m^3/s
        volume: float  # m^3

    def __init__(self, species: Species):
        self.species_models = {s: GenericDensityModel(s) for s in species}

    def __call__(self, state: State, params: Params) -> State:
        def calc_single_species(species: Species) -> GenericDensityModel.State:
            """Evaluate the model for a single species.

            Args:
                species (Species): species to evaluate the model for.

            Returns:
                GenericDensityModel.State: state derivative for the species.
            """
            model = self.species_models[species]
            state_dot, debugs = model(
                GenericDensityModel.State(volume_average_density=state.vol_avg_ion[species]),
                GenericDensityModel.Params(
                    sources_and_sinks=params.sources_and_sinks[species],
                    species_confinement_time=params.species_confinement_time[species],
                    volume_dot=params.volume_dot,
                    volume=params.volume,
                ),
            )
            return state_dot.volume_average_density, debugs

        state_dot_and_debug = {species: calc_single_species(species) for species in self.species_models.keys()}
        state_dot = {species: state_dot for species, (state_dot, _) in state_dot_and_debug.items()}
        debugs = {species: debugs for species, (_, debugs) in state_dot_and_debug.items()}
        return MultiSpeciesDensityModel.State(vol_avg_ion=state_dot), debugs


def zeff_term(charge_state: float, species_density: float, electron_density: float) -> float:
    return (charge_state**2 * species_density) / electron_density


class DensityModelImpurityCalc(eqx.Module):
    """
    Calculator to map a MultiSpeciesDensityModel.State and average pressure to effective charge, dilution, and volume-averaged electron density.
    The key assumptions are:
        1) Impurity charge states are insensitive to order 10% variations in the electron density (see unit test "test_sensitivity_to_electron_density").
        2) Fuel is fully ionized.
        3) RADAS curves are evaluated using volume-averages.
    """

    fuel_species: Sequence[FuelSpecies]
    impurity_species: Sequence[Impurity]
    mean_charge_curves: dict[Impurity, Callable[[float], float]]
    Lz_curves: dict[Impurity, Callable[[float], float]]

    def __init__(self, species: Species):
        self.fuel_species = [s for s in species if isinstance(s, FuelSpecies)]
        self.impurity_species = [s for s in species if isinstance(s, Impurity)]
        atomic_data = read_atomic_data()
        self.mean_charge_curves = {impurity: atomic_data[impurity].coronal_mean_Z_interpolator for impurity in self.impurity_species}
        self.Lz_curves = {impurity: atomic_data[impurity].coronal_Lz_interpolator for impurity in self.impurity_species}

    def __call__(self, density_state: MultiSpeciesDensityModel.State, average_pressure_kev_1e19: float) -> dict[str, float]:
        """Calculate the effective charge, dilution, and volume-averaged electron density given
        particle densities and the average electron pressure.

        Args:
            density_state (MultiSpeciesDensityModel.State): state of the MultiSpeciesDensityModel.
            average_pressure_kev_1e19 (float): average pressure in keV-1e19.

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
        imp_charge_states = {
            impurity: impurity_effects.calc_impurity_charge_state_impl(
                1e19 * average_electron_density_19_guess, 1000 * average_electron_temp_kev_guess, self.mean_charge_curves[impurity]
            )
            for impurity in self.impurity_species
        }

        # Now that we have the charge states, we can calculate the electron density contributions from each impurity species.
        electron_19_from_imp = jnp.sum(
            jnp.array([charge_state * density_state.vol_avg_ion[imp] for imp, charge_state in imp_charge_states.items()])
        )

        # Do the same for the fuel. Assume full ionization.
        electron_19_from_fuel = jnp.sum(
            jnp.array([AtomicNumberMap[species] * density_state.vol_avg_ion[species] for species in self.fuel_species])
        )

        # Add the contributions together to get the total electron density.
        electron_density_19 = electron_19_from_imp + electron_19_from_fuel

        # Calculate the dilution.
        dilution = density_state.total_volume_average_ion_density / electron_density_19

        # Compute the z-effective terms for each species.
        zeff_terms = {
            species: zeff_term(AtomicNumberMap[species], density_state.vol_avg_ion[species], electron_density_19)
            for species in density_state.vol_avg_ion.keys()
        }

        impurity_concentrations = {
            impurity: density_state.vol_avg_ion[impurity] / electron_density_19 for impurity in self.impurity_species
        }

        outs = {
            "z_effective": jnp.sum(leaves_as_array(zeff_terms)),
            "dilution": dilution,
            "volume_average_electron_density_19": electron_density_19,
            "impurity_concentrations": impurity_concentrations,
        }
        return outs

    def calc_impurity_radiated_power_radas(
        self,
        electron_temp_profile,
        electron_density_profile,
        impurity_concentrations,
        volume_integrator,
    ):
        components = {}
        for impurity, concentration in impurity_concentrations.items():
            components[impurity] = radiated_power.calc_impurity_radiated_power_radas(
                electron_temp_profile=electron_temp_profile,
                electron_density_profile=electron_density_profile,
                impurity_concentration=concentration,
                volume_integrator=volume_integrator,
                Lz_curve=self.Lz_curves[impurity],
            )
        values = jnp.array(list(components.values()))
        debug = components
        return jnp.sum(values), debug
