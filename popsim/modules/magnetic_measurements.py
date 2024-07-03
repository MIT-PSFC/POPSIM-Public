import chex

import numpy as np

from popsim import ModuleBase
from popsim.modules.tearing import Tearing, Island

"""
Classes for making magnetic measurments.
"""

def build_design_matrix(probe_connections: list[tuple[float, float]]) -> np.ndarray:
    """Build a design matrix from the probe connections.

    Args:
        probe_connections (list[tuple[float, float]]): A list of tuples representing the connections between probes.

    Returns:
        np.ndarray: The design matrix.
    """
    

@chex.dataclass
class LowNArray(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.
        #phi_probes: list[float] # I don't think this is needed, information is captured in connections
        probe_connections: list[tuple[float, float]]

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        tearing_state: Tearing.State

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        reconstructed_magnitudes: dict[Island, float]

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        #func_Bp_per_A: function
        #func_Br_per_A: function
        func_Bp_per_A: float
        tearing_params: Tearing.Params

    config: Config
    tearing_module: Tearing
    design_matrix: np.ndarray

    def __init__(self, config, tearing_module: Tearing):
        self.config = config
        self.tearing_module = tearing_module

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:

        tearing_dot, tearing_out = self.tearing_module(state.tearing_state, params.tearing_params)

        # Get the perturbed current, phase, and frequency of each tearing mode
        mode_currents = tearing_out.mode_current
        mode_phases = tearing_dot.mode_phase
        mode_freqs = tearing_dot.F

        # Reconstruct the magnitudes of the tearing modes
        reconstructed_magnitudes = {}

        # Make a state_dot.
        state_dot = LowNArray.State(
            tearing_state=tearing_dot,
        )
        # Make an output.
        out = LowNArray.Output(
            reconstructed_magnitudes=reconstructed_magnitudes,
        )
        return state_dot, out
