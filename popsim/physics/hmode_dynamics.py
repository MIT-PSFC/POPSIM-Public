import chex
import equinox as eqx
import jax.numpy as jnp

"""
An implementation of Hmode dynamics using a continuous differential equations model.
Note that the main foot-gun of this model is that the value of h-mode can go above 1.0 or below 0.0.
The clipping in __init__ doesn't seem to help when doing a diffrax.diffeqsolve.
"""

CRITICAL_THRESHOLD = 0.5


class State(eqx.Module):
    hmode: chex.Numeric

    def __init__(self, hmode: chex.Numeric, is_derivative: bool = False):
        self.hmode = jnp.where(is_derivative, hmode, jnp.clip(hmode, 0.0, 1.0))

    @property
    def in_hmode(self):
        return jnp.bool_(self.hmode >= CRITICAL_THRESHOLD)

    @property
    def max_value(self):
        return 1.0

    @property
    def min_value(self):
        return 0.0


@chex.dataclass
class Params:
    transition_characteristic_time: chex.Numeric  # amount of time for a L->H or H->L transition to occur [s].
    P_tau_MW: chex.Numeric  # Power conducted to the scrape-off layer [MW]
    P_input_MW: chex.Numeric  # Power input to the plasma [MW]
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

    # If we are in H-mode, the relevant threshold is the one for H->L transition.
    # If we are in L-mode, the relevant threshold is the one for L->H transition.
    threshold = jnp.where(state.in_hmode, params.hl_threshold_MW, params.lh_threshold_MW)

    # If we want the transition from L-mode to H-mode to happen in 0.1s, then we need to get
    # from 0 to 0.5 in 0.1s, which means we need a speed of 0.5 / 0.1 = 5.
    # Similarly for H->L transition.
    transition_speed = CRITICAL_THRESHOLD / params.transition_characteristic_time

    # Trigger L->H if both conducted power and input power are above the threshold.
    # I originally just had a cnoducted power condition, but in H->L back transitions
    # the conducted power shoots up for a while, which causes a trigger back to H mode which doesn't make sense.
    trigger_l_to_h = jnp.logical_and(params.P_tau_MW >= threshold, params.P_input_MW >= threshold)
    hmode_dot = jnp.where(trigger_l_to_h, transition_speed, -transition_speed)

    # If hmode_dot > 0.0 but we the state is already at max, then we should set hmode_dot to 0.0.
    hmode_dot = jnp.where(jnp.logical_and(hmode_dot > 0.0, state.hmode >= state.max_value), 0.0, hmode_dot)

    # If hmode_dot < 0.0 but we the state is already at min, then we should set hmode_dot to 0.0.
    hmode_dot = jnp.where(jnp.logical_and(hmode_dot < 0.0, state.hmode <= state.min_value), 0.0, hmode_dot)

    return State(hmode=hmode_dot, is_derivative=True)
