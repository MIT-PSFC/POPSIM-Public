from collections.abc import Sequence

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from popsim import ModuleBase
from popsim.enums import Species
from popsim.tree_util import leaves_as_array


class Density(ModuleBase):
    """A multi-species density dynamics model that evolves volume-averaged ion densities for each species."""

    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class Output:
        pass

    @chex.dataclass
    class State:
        vol_avg_ion: dict[Species, float]  # 1e19/m^3

        @property
        def total_volume_average_ion_density(self) -> float:
            # Logic to make sure this works for both the scalar and array cases.
            # See associated unit test.
            leaves = jax.tree_util.tree_leaves(self.vol_avg_ion)
            leaves = jax.tree.map(jnp.atleast_1d, leaves)  # Promote scalars.
            n_t = leaves[0].size  # Number of time steps.
            leaves = eqx.error_if(leaves, any(leaf.size != n_t for leaf in leaves), "All species must have the same size.")
            if n_t > 1:
                # Do a separate computation for each time step.
                return jax.vmap(lambda x: jnp.sum(leaves_as_array(x)))(self.vol_avg_ion)
            else:
                # Do a single computation.
                return jnp.sum(leaves_as_array(self.vol_avg_ion))

        @property
        def species(self) -> Sequence[Species]:
            return list(self.vol_avg_ion.keys())

    @chex.dataclass
    class Inputs:
        sources_and_sinks: dict[Species, PyTree[float]]  # PyTree of net particle fluxes from various sources and sinks 1e19/s
        species_confinement_time: dict[Species, float]  # species confinement time in seconds
        volume_dot: float  # m^3/s
        volume: float  # m^3

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        return multi_species_derivs(state, inputs), Density.Output()


def single_species_derivs(
    volume_average_density: float, volume: float, volume_dot: float, species_confinement_time: float, sources_and_sinks: PyTree[float]
) -> float:
    """Compute the time derivative of the volume-averaged ion density for a single species.

    Args:
        volume_average_density (float): volume average ion density [1/X^3] for any length X.
        volume (float): volume of the plasma [X^3] for any length X.
        volume_dot (float): rate of change of the volume [X^3/s] for any length X.
        species_confinement_time (float): confinement time of the species [s].
        sources_and_sinks (PyTree[float]): net particle fluxes from various sources and sinks [1/s].

    Returns:
        float: time derivative of the volume-averaged ion density [1/X^3/s] for any length X.
    """
    N = volume * volume_average_density

    sources_and_sinks = jnp.array(jax.tree_util.tree_leaves(sources_and_sinks))
    net_particle_flux = jnp.sum(sources_and_sinks)
    N_dot = -N / species_confinement_time + net_particle_flux

    # By the quotient rule:
    n_dot = (N_dot * volume - N * volume_dot) / volume**2
    return n_dot


def multi_species_derivs(state: Density.State, inputs: Density.Inputs) -> Density.State:
    """Compute the time derivative of the volume-averaged ion densities for all species.

    Args:
        state (State): density state of the plasma.
        inputs (Inputs): external inputs.

    Returns:
        State: time derivative of the density state of the plasma.
    """

    def calc_single_species(species: Species) -> float:
        state_dot = single_species_derivs(
            state.vol_avg_ion[species],
            sources_and_sinks=inputs.sources_and_sinks[species],
            species_confinement_time=inputs.species_confinement_time[species],
            volume_dot=inputs.volume_dot,
            volume=inputs.volume,
        )
        return state_dot

    state_dot = {species: calc_single_species(species) for species in state.species}
    return Density.State(vol_avg_ion=state_dot)
