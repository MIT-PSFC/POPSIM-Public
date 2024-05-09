from popsim.modules.power_balance import PowerBalance
import jax.numpy as jnp
import jax

def test_dynamics():
    """ Test that the power balance dynamics are properly calculated
    """

    # Set up confinement times

    tau_E = 1.0 #Energy confinement time [s]

    # Set initial state
    state = PowerBalance.State(
        stored_energy=1.0,
    )

    # Set params

    params = PowerBalance.Params(
        P_aux=1.0,  # Auxillary heating power [MW]
        confinement_time=tau_E,  # energy confinement time in seconds
    )

    power_balance_module = PowerBalance(config=PowerBalance.Config())

    power_balance_dot, power_balance_output = power_balance_module(state, params)

    assert power_balance_dot.stored_energy == 0.0