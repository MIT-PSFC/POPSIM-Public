import diffrax
import equinox as eqx
import jax.numpy as jnp

from popsim.simulators.comet_mirror.build_default import build_default
from popsim.simulators.comet_mirror.integrator import Integrator


def test_comet_mirror():
    # Test that the simulator runs.
    model, state, params = build_default()
    state_dot = model(state, params)
    sim = Integrator(model)
    ts = jnp.linspace(0, 0.1, 10)

    current = params.plasma_current + -1.0e6 * ts
    current_intrep = diffrax.CubicInterpolation(
        ts=ts,
        coeffs=diffrax.backward_hermite_coefficients(ts, current),
    )
    params = eqx.tree_at(where=lambda p: p.plasma_current, pytree=params, replace=current_intrep)
    sol, debugs = sim(ts, state, params, debug_info=True)