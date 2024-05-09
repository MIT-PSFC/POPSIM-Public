from popsim.modules.icrh_zone import IcrhZone
from popsim.enums import FuelSpecies, Impurity, SpeciesContainer
import jax.numpy as jnp
import jax

def test_dynamics():
    """ Test that the dynamics are properly calculated
    """

    # Set initial state
    state = IcrhZone.State()

    # Set params
    params = IcrhZone.Params(
        frequency_command=120,
        power_command=2e6,
    )

    icrh_zone_module = IcrhZone(config=IcrhZone.Config())
    
    icrh_zone_dot, icrh_zone_output = icrh_zone_module(state, params)

    assert icrh_zone_output.reflected_power == 0.2e6
    assert icrh_zone_output.transmitted_power == 1.8e6