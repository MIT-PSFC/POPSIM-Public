import chex
from jaxtyping import ArrayLike, PyTree

from popsim import ModuleBase
from popsim.modules.magnetic_diagnostics import LowNArray
from popsim.modules.tearing import Tearing

"""
Simulation for multiple tearing modes and all the diagnostics which can measure them.
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
        tearing_state: Tearing.State

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        locals: PyTree[ArrayLike]

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        tearing_params: Tearing.Params

    config: Config
    tearing_module: Tearing
    low_n_array_module: LowNArray

    def __init__(self, config: Config, tearing_module: Tearing, low_n_array_module: LowNArray):
        self.config = config


    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        # Make a state_dot.
        state_dot = ExampleTemplate.State()
        # Make an output.
        out = ExampleTemplate.Output()
        return state_dot, out
