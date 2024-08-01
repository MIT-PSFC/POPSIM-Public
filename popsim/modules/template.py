import chex
import jax

from popsim import ModuleBase

"""
An example template to copy and paste when creating a new module.
"""


@chex.dataclass
class ExampleTemplate(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.
        pass

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        pass

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        pass

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        pass

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params, key: jax.random.PRNGKey = None) -> tuple[State, Output]:
        # Make a state_dot.
        state_dot = ExampleTemplate.State()
        # Make an output.
        out = ExampleTemplate.Output()
        return state_dot, out
