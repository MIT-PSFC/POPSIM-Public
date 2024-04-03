import chex
import jax.numpy as jnp

"""
An implementation of Hmode dynamics using a continuous differential equations model.
"""


@chex.dataclass
class State:
    hmode: chex.Numeric
    CRITICAL_THRESHOLD: chex.Numeric = 0.5

    @property
    def in_hmode(self):
        return jnp.bool_(self.hmode >= self.CRITICAL_THRESHOLD)

    def in_bounds(self, x):
        return jnp.logical_and(x >= 0, x <= 1)


@chex.dataclass
class Params:
    transition_characteristic_time: chex.Numeric  # amount of time for a L->H or H->L transition to occur [s].
    P_tau_MW: chex.Numeric  # Power conducted to the scrape-off layer [MW]
    hl_threshold_MW: chex.Numeric  # threshold for the H-mode to L-mode transition [MW].
    lh_threshold_MW: chex.Numeric  # threshold for the L-mode to H-mode transition [MW].


def dynamics(state: State, params: Params) -> State:
    """Compute the time derivative of the H-mode state.

    Args:
        state (State): current state.
        params (Params): input parameters.

    Returns:
        State: time derivative of the H-mode state.
    """
    threshold = jnp.where(state.in_hmode, params.hl_threshold_MW, params.lh_threshold_MW)

    # If we want the transition from L-mode to H-mode to happen in 0.1s, then we need to get
    # from 0 to 0.5 in 0.1s, which means we need a speed of 0.5 / 0.1 = 5.
    # Similarly for H->L transition.
    transition_speed = state.CRITICAL_THRESHOLD / params.transition_characteristic_time
    hmode_dot = jnp.where(params.P_tau_MW < threshold, -transition_speed, transition_speed)

    hmode_dot = jnp.where(state.in_bounds(state.hmode), hmode_dot, 0.0)

    return State(hmode=hmode_dot)
