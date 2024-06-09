from enum import IntEnum

import chex
import jax.numpy as jnp

from popsim import ModuleBase

"""
An example discrete module.
"""


class ExampleState(IntEnum):
    """ """

    Rotating = 0
    Decelerating = 1
    Locked = 2


@chex.dataclass
class DiscreteExample(ModuleBase):
    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        example_state: ExampleState
        example_cont: float = 0.0

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        pass

    @chex.dataclass
    class Params:
        time: float
        time_to_decel: float
        time_to_lock: float

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        # Update the state
        example_state = jnp.where(params.time > params.time_to_decel, ExampleState.Decelerating, ExampleState.Rotating)

        example_state = jnp.where(params.time > params.time_to_lock, ExampleState.Locked, example_state)

        state_new = DiscreteExample.State(example_state=example_state, example_cont=-state.example_cont)
        # Make an output.
        out = DiscreteExample.Output()
        return state_new, out
