import jax.numpy as jnp

from popsim.simulators.comet_mirror.build_default import build_default
from popsim.simulators.comet_mirror.integrator import Integrator


def test_comet_mirror():
    # Test that the simulator runs.
    model, state, params = build_default()
    state_dot = model(state, params)
    sim = Integrator(model)
    ts = jnp.linspace(0, 0.1, 10)
    sol, debugs = sim(ts, state, params, debug_info=True)