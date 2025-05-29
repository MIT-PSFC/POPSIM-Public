import chex
import jax.numpy as jnp

from popsim import TimeDepModule, discrete_time_field


class SignalRMS(TimeDepModule):
    """Calculates the RMS of a signal over a moving window."""

    @chex.dataclass
    class State:
        window: jnp.ndarray = discrete_time_field(default_factory=lambda: jnp.array([]))

    @chex.dataclass
    class Output:
        rms: float

    @chex.dataclass
    class Inputs:
        signal: float

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        # Add the new signal to the window at the current index and update the index.
        sig1d = jnp.atleast_1d(inputs.signal)
        new_window = jnp.concatenate((sig1d, state.window[:-1]))

        rms = jnp.sqrt(jnp.mean(new_window**2))

        state_out = SignalRMS.State(window=new_window)
        output = SignalRMS.Output(rms=rms)

        return state_out, output
