import chex
import jax.numpy as jnp

from popsim import ModuleBase, discrete_time_field


@chex.dataclass
class SignalRMS(ModuleBase):
    """Calculates the RMS of a signal over a moving window."""

    @chex.dataclass
    class Config:
        window_size: int

    @chex.dataclass
    class State:
        window: jnp.ndarray = discrete_time_field(default_factory=lambda: jnp.array([]))
        index: int = discrete_time_field(default=0)

    @chex.dataclass
    class Output:
        rms: float

    @chex.dataclass
    class Params:
        signal: float

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        # Add the new signal to the window at the current index and update the index.
        new_window = state.window.at[state.index].set(params.signal)
        new_index = (state.index + 1) % self.config.window_size

        rms = jnp.sqrt(jnp.mean(new_window**2))

        state_out = SignalRMS.State(window=new_window, index=new_index)
        output = SignalRMS.Output(rms=rms)

        return state_out, output
