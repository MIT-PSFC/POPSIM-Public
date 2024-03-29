import diffrax
import equinox as eqx
import jax.numpy as jnp

from popsim.scenarios.sparc_prd.comet_mirror import build_comet_mirror_config
from popsim.simulators.comet_mirror.simulate import simulate

def test_comet_mirror():
    # Test that the simulator runs.
    model, state, params = build_comet_mirror_config()
    state_dot = model(state, params)
    ts = jnp.linspace(0, 0.1, 10)

    current = params.plasma_current + -1.0e6 * ts
    current_intrep = diffrax.CubicInterpolation(
        ts=ts,
        coeffs=diffrax.backward_hermite_coefficients(ts, current),
    )
    params.plasma_current = current_intrep

    sol = simulate(model, ts, state, params)