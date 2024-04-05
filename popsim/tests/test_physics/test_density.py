import popsim.physics.density as density
from popsim.enums import FuelSpecies, Impurity
import jax.numpy as jnp

def test_state():
    #
    # Main goal here is to test that the 
    #
    state = density.State(
        vol_avg_ion={FuelSpecies.Deuterium: 1.0, Impurity.Tungsten: 2.0}
    )
    assert state.total_volume_average_ion_density == 3.0

    state = density.State(
        vol_avg_ion={FuelSpecies.Deuterium: jnp.array(1.0), Impurity.Tungsten: jnp.array(2.0)}
    )

    assert state.total_volume_average_ion_density == 3.0

    state = density.State(
        vol_avg_ion={FuelSpecies.Deuterium: jnp.array([1.0, 2.0]), Impurity.Tungsten: jnp.array([2.0, 3.0])}
    )
    assert (state.total_volume_average_ion_density == jnp.array([3.0, 5.0])).all()