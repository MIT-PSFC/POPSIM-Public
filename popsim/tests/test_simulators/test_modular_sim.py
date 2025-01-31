import jax
import jax.numpy as jnp

import popsim.simulators.modular_sim.model as msm
from popsim.simulators.modular_sim.model import ModularModel
from popsim.simulate import simulate, SimInput
from popsim.enums import SpeciesContainer, FuelSpecies, Impurity

def test_modular_sim():
    # Configure the model
    species = [FuelSpecies.Deuterium, FuelSpecies.Tritium, Impurity.Helium, Impurity.Tungsten]
    species_container = SpeciesContainer(species=species)

    config = ModularModel.Config(
        species=species_container,
        icrh_zone=msm.IcrhZone.Config(),
        finj_injectors = {k:msm.FinjInjector.Config() for k in species_container.species},
        power_balance = msm.PowerBalance.Config(),
        density = msm.Density.Config(),
    )

    model = msm.ModularModel(config=config)

    # Set inputs
    particle_confinement_scalars = {
        k: 3.0 if k in species_container.fuel_species else 10.0
        for k in species_container.species
    }

    inputs = ModularModel.Inputs(
        confinement_time_scalar=1.0,
        confinement_time=1.0,
        P_aux_MW=1.0,
        fueling19={
            FuelSpecies.Deuterium: 1.0,
            FuelSpecies.Tritium: 0.0,
            Impurity.Helium: 1.0, 
            Impurity.Tungsten: 0.0},
        particle_confinement_scalar=particle_confinement_scalars,
        valve_flow_rate_tau= 1.0,
        pipe_flow_rate_tau=1.0,
        volume = 13.0,
        volume_dot = 0.0,
    )

    # Set initial state
    state = ModularModel.State(
        power_balance_state=msm.PowerBalance.State(stored_energy=1.0),
        density_state=msm.Density.State(vol_avg_ion={
            FuelSpecies.Deuterium: 1.0,
            FuelSpecies.Tritium: 0.0,
            Impurity.Helium: 1.0, 
            Impurity.Tungsten: 0.0}),
        icrh_zone_state=msm.IcrhZone.State(),
        finj_state={k:msm.FinjInjector.State(valve_flow_rate=0.0, pipe_flow_rate=0.0) for k in species_container.species},
    )

    # Simulate the model
    ts = jnp.linspace(0, 1.0, 10)
    sol = simulate(model, SimInput(time=ts, initial_state=state,inputs=inputs), return_xarray=True)