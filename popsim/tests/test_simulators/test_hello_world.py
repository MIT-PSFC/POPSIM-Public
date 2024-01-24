import jax.numpy as jnp
from popsim.simulators.simple_power_balance.build_default import build_default
from popsim.simulators.simple_power_balance.simple_model import Simulator

def test_hello_world():
    model, state, params = build_default()
    dW_dt, debug = model(state, params, debug_info=True)

    simulate = Simulator(model)
    ts = jnp.linspace(0.0, 2.0, 100)
    sol, derivs, debugs = simulate(ts, state, params)
