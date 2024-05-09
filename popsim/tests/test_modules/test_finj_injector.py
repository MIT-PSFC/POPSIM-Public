from popsim.modules.finj_injector import FinjInjector
import jax.numpy as jnp
import jax

def test_dynamics():
    """ Test that the FINJ injector dynamics are properly calculated
    """

    # Set up time constants

    valve_flow_rate_tau = 2.0 # valve flow rate confinement time [s]
    pipe_flow_rate_tau = 3.0 # pipe flow rate confinement time [s]

    # Set initial state
    state = FinjInjector.State(
        valve_flow_rate=2.0,
        pipe_flow_rate=1.0,
    )

    # Set params
    params = FinjInjector.Params(
        flow_rate_command = 3.0, # [Pa m^3/s]
        valve_flow_rate_tau = valve_flow_rate_tau, # [s]
        pipe_flow_rate_tau = pipe_flow_rate_tau # [s]
    )

    finj_injector_module = FinjInjector(config=FinjInjector.Config())

    finj_injector_dot, finj_injector_output = finj_injector_module(state, params)

    assert finj_injector_output.valve_flow_rate == 2.0
    assert finj_injector_dot.valve_flow_rate == 1.0/2.0
    assert finj_injector_dot.pipe_flow_rate == 1.0/3.0