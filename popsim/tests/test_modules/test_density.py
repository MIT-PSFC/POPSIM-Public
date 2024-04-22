from popsim.modules.density import Density
from popsim.enums import FuelSpecies, Impurity, SpeciesContainer
import jax.numpy as jnp
import jax

def test_state():
    """ The main goal here is to test the total ion density function 
    """

    state = Density.State(
        vol_avg_ion={FuelSpecies.Deuterium: 1.0, Impurity.Tungsten: 2.0}
    )
    assert state.total_volume_average_ion_density == 3.0

    state = Density.State(
        vol_avg_ion={FuelSpecies.Deuterium: jnp.array(1.0), Impurity.Tungsten: jnp.array(2.0)}
    )

    assert state.total_volume_average_ion_density == 3.0

    state = Density.State(
        vol_avg_ion={FuelSpecies.Deuterium: jnp.array([1.0, 2.0]), Impurity.Tungsten: jnp.array([2.0, 3.0])}
    )
    assert (state.total_volume_average_ion_density == jnp.array([3.0, 5.0])).all()

def test_dynamics():
    """ Test that the dynamics are properly calculated
    """
    species = [FuelSpecies.Deuterium, FuelSpecies.Tritium, Impurity.Helium, Impurity.Tungsten]
    species_container = SpeciesContainer(species=species)

    volume = 13.0 # Plasma volume [m^3]

    # Set up confinement times
    additional_assumptions = {
        "k_fuel": 2.0,  # Particle confinement scalar for fuel species.
        "k_impurity": 10.0,  # Particle confinement scalar for impurity species.
    }
 
    tau_E = 1.0 #Energy confinement time [s]
    particle_confinement_scalars = {
        k: additional_assumptions["k_fuel"] if k in species_container.fuel_species else additional_assumptions["k_impurity"]
        for k in species_container.species
    }

    # Set up sources and sinks
    sources_and_sinks = {k: {} for k in species}
    for k in species:
        sources_and_sinks[k]["fueling19"] = volume

    sources_and_sinks[FuelSpecies.Deuterium]["fusion"] = -volume
    sources_and_sinks[FuelSpecies.Tritium]["fusion"] = -volume
    sources_and_sinks[Impurity.Helium]["fusion"] = volume

    # Set initial state
    state = Density.State(
        vol_avg_ion={
            FuelSpecies.Deuterium: 1.0,
            FuelSpecies.Tritium: 0.0,
            Impurity.Helium: 1.0, 
            Impurity.Tungsten: 0.0}
    )

    # Set params
    params = Density.Params(
        sources_and_sinks=sources_and_sinks,
        species_confinement_time=jax.tree_map(lambda k: k * tau_E, particle_confinement_scalars),
        volume_dot=0.0,
        volume=13,
    )

    density_module = Density(config=Density.Config())
    
    state_dot, output = density_module(state, params)

    # Check state_dot
    assert state_dot.vol_avg_ion[FuelSpecies.Deuterium] == -0.5
    assert state_dot.vol_avg_ion[FuelSpecies.Tritium] == -0.0
    assert jnp.isclose(state_dot.vol_avg_ion[Impurity.Helium],1.9)
    assert state_dot.vol_avg_ion[Impurity.Tungsten] == 1.0

    # Repeat with non-zero volume_dot:
    # Set params
    params = Density.Params(
        sources_and_sinks=sources_and_sinks,
        species_confinement_time=jax.tree_map(lambda k: k * tau_E, particle_confinement_scalars),
        volume_dot=13.0,
        volume=13,
    )

    state_dot, output = density_module(state, params)

    # Check state_dot
    assert state_dot.vol_avg_ion[FuelSpecies.Deuterium] == -1.5
    assert state_dot.vol_avg_ion[FuelSpecies.Tritium] == -0.0
    assert jnp.isclose(state_dot.vol_avg_ion[Impurity.Helium],0.9)
    assert state_dot.vol_avg_ion[Impurity.Tungsten] == 1.0
         