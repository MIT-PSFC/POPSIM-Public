import chex
import jax.numpy as jnp
from jaxtyping import Array, ScalarLike

from popsim import TimeDepModule, discrete_time_field


class DelayBuffer(TimeDepModule):
    """A simple delay buffer that delays the input by a number of time steps equal to the buffer size."""

    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        buffer: Array = discrete_time_field()

    @chex.dataclass
    class Output:
        delayed: ScalarLike

    @chex.dataclass
    class Inputs:
        inp: ScalarLike

    config: Config

    def __init__(self, config=None):
        self.config = config or self.Config()

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        # The first element of the buffer is the delayed input.
        output = DelayBuffer.Output(delayed=state.buffer[0])

        # Essentially shift the buffer by one and insert the new input at the end.
        state_out = DelayBuffer.State(buffer=jnp.concatenate([state.buffer[1:], jnp.array([inputs.inp])]))
        return state_out, output
