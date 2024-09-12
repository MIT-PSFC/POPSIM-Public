import dataclasses

import chex
from jaxtyping import ArrayLike, PyTree

from popsim import ModuleBase
from popsim.modules.magnetic_diagnostics import BFieldPoloidalProbes, LowNArray
from popsim.modules.tearing import Tearing

"""
Simulation for tearing modes and all the diagnostics which can measure them.
"""


@chex.dataclass
class TearingSim(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.
        tearing_module: Tearing = dataclasses.field(default_factory=lambda: Tearing())
        b_field_poloidal_probes_module: BFieldPoloidalProbes
        lown_array_module: LowNArray

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

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        # Compute the results of the Tearing module.
        tearing_state_dot, tearing_out = self.config.tearing_module(state.tearing_state, params.tearing_params)

        # Make a state_dot.
        state_dot = TearingSim.State(tearing_state=tearing_state_dot)

        # Build the parameters for the LowNArray module.
        lown_array_params = LowNArray.Params(tearing_out=tearing_out, modes=self.config.tearing_module.config.modes)

        lown_array_out = self.config.lown_array_module(None, lown_array_params)
        out = TearingSim.Output(locals=lown_array_out)
        return state_dot, out
