import chex
import jax.numpy as jnp

from popsim import ModuleBase

CRITICAL_THRESHOLD = 0.5


@chex.dataclass
class HmodeDynamics(ModuleBase):
    """An H-Mode dynamics module.

    An implementation of Hmode dynamics using a low-pass filter approach where:
            hmode_dot = (hmode_pass - hmode) / transition_characteristic_time
    and hmode_pass is 1 if the transition condition is met and 0 otherwise. If h_mode >= 0.5, we are in H-mode, otherwise we are in L-mode.
    """

    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class Output:
        pass

    @chex.dataclass
    class State:
        hmode: chex.Numeric

        @property
        def in_hmode(self):
            return jnp.bool_(self.hmode >= CRITICAL_THRESHOLD)

    @chex.dataclass
    class Params:
        transition_characteristic_time: chex.Numeric  # amount of time for a L->H or H->L transition to occur [s].
        P_tau_MW: chex.Numeric  # Power conducted to the scrape-off layer [MW]
        P_input_MW: chex.Numeric  # Power input to the plasma [MW]
        hl_threshold_MW: chex.Numeric  # threshold for the H-mode to L-mode transition [MW].
        lh_threshold_MW: chex.Numeric  # threshold for the L-mode to H-mode transition [MW].

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
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

        # Trigger L->H if both conducted power and input power are above the threshold.
        # I originally just had a conducted power condition, but in H->L back transitions
        # the conducted power shoots up for a while, which causes a trigger back to H mode which doesn't make sense.
        trigger_l_to_h = jnp.logical_and(params.P_tau_MW >= threshold, params.P_input_MW >= threshold)
        hmode_pass = jnp.where(trigger_l_to_h, 1.0, 0.0)

        hmode_dot = (hmode_pass - state.hmode) / params.transition_characteristic_time
        return HmodeDynamics.State(hmode=hmode_dot), HmodeDynamics.Output()


# TODO: This is only here to keep cometmirror working for now
def dynamics(state: HmodeDynamics.State, params: HmodeDynamics.Params) -> HmodeDynamics.State:
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

    # Trigger L->H if both conducted power and input power are above the threshold.
    # I originally just had a conducted power condition, but in H->L back transitions
    # the conducted power shoots up for a while, which causes a trigger back to H mode which doesn't make sense.
    trigger_l_to_h = jnp.logical_and(params.P_tau_MW >= threshold, params.P_input_MW >= threshold)
    hmode_pass = jnp.where(trigger_l_to_h, 1.0, 0.0)

    hmode_dot = (hmode_pass - state.hmode) / params.transition_characteristic_time
    return HmodeDynamics.State(hmode=hmode_dot)
