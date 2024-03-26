from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from popsim.enums import Species
from popsim.tree_util import leaves_as_array


@chex.dataclass
class State:
    vol_avg_ion: dict[Species, float]  # 1e19/m^3

    @property
    def total_volume_average_ion_density(self) -> float:
        return jnp.sum(leaves_as_array(self.vol_avg_ion))

    @property
    def species(self) -> Sequence[Species]:
        return list(self.vol_avg_ion.keys())


@chex.dataclass
class Params:
    sources_and_sinks: dict[Species, PyTree[float]]  # PyTree of net particle fluxes from various sources and sinks 1e19/s
    species_confinement_time: dict[Species, float]  # species confinement time in seconds
    volume_dot: float  # m^3/s
    volume: float  # m^3


def single_species_derivs(
    volume_average_density: float, volume: float, volume_dot: float, species_confinement_time: float, sources_and_sinks: PyTree[float]
) -> float:
    N = volume * volume_average_density

    sources_and_sinks = jnp.array(jax.tree_util.tree_leaves(sources_and_sinks))
    net_particle_flux = jnp.sum(sources_and_sinks)
    N_dot = -N / species_confinement_time + net_particle_flux

    # By the quotient rule:
    n_dot = (N_dot * volume - N * volume_dot) / volume**2
    return n_dot


def multi_species_derivs(state: State, params: Params) -> State:
    def calc_single_species(species: Species) -> State:
        """Evaluate the model for a single species.

        Args:
            species (Species): species to evaluate the model for.

        Returns:
            State: state derivative for the species.
        """
        state_dot = single_species_derivs(
            state.vol_avg_ion[species],
            sources_and_sinks=params.sources_and_sinks[species],
            species_confinement_time=params.species_confinement_time[species],
            volume_dot=params.volume_dot,
            volume=params.volume,
        )
        return state_dot

    state_dot = {species: calc_single_species(species) for species in state.species}
    return State(vol_avg_ion=state_dot)
