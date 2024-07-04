import chex

import numpy as np
from scipy.interpolate import interp1d

from popsim import ModuleBase
from popsim.modules.tearing import Tearing, Island

"""
Classes for making magnetic measurments.
"""

def build_design_matrix(probe_connections: list[tuple[float, float]], unique_mode_numbers: np.ndarray) -> np.ndarray:
    """Build the design matrix that encodes the connections between probes and the mode numbers of the tearing modes being measured.

    Args:
        probe_connections (list[tuple[float, float]]): A list of tuples representing the connections between probes.
        unique_mode_numbers (np.ndarray): The mode numbers of the tearing modes being measured.

    Returns:
        np.ndarray: The design matrix.
    """
    
    design_matrix = np.zeros((len(probe_connections), 2*len(unique_mode_numbers)))

    for i, (probe1_angle, probe2_angle) in enumerate(probe_connections):
        for j, mode_number in enumerate(unique_mode_numbers):
            design_matrix[i, 2*j] = np.cos(mode_number*probe1_angle) - np.cos(mode_number*probe2_angle)
            design_matrix[i, 2*j+1] = np.sin(mode_number*probe1_angle) - np.sin(mode_number*probe2_angle)

    return design_matrix

def get_differenced_signals(probe_connections: list[tuple[float, float]], filtered_mags: dict[float, float], mode_phases, islands) -> np.ndarray:
    """The Low-N array has digitizers that record the differences between probes to isolate 
    the small tearing mode signal (~0.001 T) on top of the large background field (~10 T)
    
    Args:
        probe_connections (list[tuple[float, float]]): A list of tuples representing the connections between probes.
        filtered_mags (dict[float, float]): The magnitudes of the tearing modes after being filtered by the transfer function.
        mode_phases: The phases of the tearing modes.
        islands: The tearing islands being measured.
    
    Returns:
        np.ndarray: The differenced signals.
    """
    
    differenced_signals = np.zeros(len(probe_connections))

    for island in islands:
        island_mag = filtered_mags[island]
        island_phase = mode_phases[island]
        island_mode = island.n
        for i, (probe1_angle, probe2_angle) in enumerate(probe_connections):
            probe_signal_1 = island_mag * np.cos(probe1_angle - (island_mode*island_phase))
            probe_signal_2 = island_mag * np.cos(probe2_angle - (island_mode*island_phase))
            differenced_signals[i] += probe_signal_1 - probe_signal_2

    return differenced_signals

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
        reconstructed_magnitudes: dict[int, float]

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        tearing_params: Tearing.Params

    config: Config
    tearing_module: Tearing
    pseudoinverse_matrix: np.ndarray
    func_Bp_per_A: interp1d  # TODO: figure out how to make this work in JIT
    unique_mode_numbers: np.ndarray

    def __init__(self, config, tearing_module: Tearing, func_Bp_per_A: interp1d):
        self.config = config
        self.tearing_module = tearing_module
        self.func_Bp_per_A = func_Bp_per_A

        # Make the matrix for reconstructing tearing mode magnitudes from probe measurements
        mode_numbers = [island.n for island in tearing_module.islands]
        self.unique_mode_numbers = np.unique(mode_numbers)
        design_matrix = build_design_matrix(self.config.probe_connections, self.unique_mode_numbers)
        self.pseudoinverse_matrix = np.linalg.pinv(design_matrix)

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:

        tearing_dot, tearing_out = self.tearing_module(state.tearing_state, params.tearing_params)

        # Get the perturbed current, phase, and frequency of each tearing mode
        mode_currents = tearing_out.mode_current
        mode_phases = tearing_out.mode_phase
        #mode_freqs = tearing_out.F

        # Convert to the signal that would be measured by the probes (adjusted by the Bp/A transfer function)
        # TODO: ran into a problem with the Bp/A transfer function, so for now just ignoring it
        #filtered_signals = {island: mode_currents[island]*self.func_Bp_per_A(mode_freqs[island]) for island in self.tearing_module.islands}
        filtered_signals = {island: mode_currents[island] for island in self.tearing_module.islands}

        # Get the differenced signals between each pair of probes
        differenced_signals = get_differenced_signals(self.config.probe_connections, filtered_signals, mode_phases, self.tearing_module.islands)

        reconstructed_components = self.pseudoinverse_matrix @ differenced_signals

        # Reconstruct the magnitudes of the tearing modes
        reconstructed_magnitudes = {}
        for i, mode_number in enumerate(self.unique_mode_numbers):
            reconstructed_magnitudes[mode_number] = np.sqrt(reconstructed_components[2*i]**2 + reconstructed_components[2*i+1]**2)

        # Make a state_dot.
        state_dot = LowNArray.State(
            tearing_state=tearing_dot,
        )
        # Make an output.
        out = LowNArray.Output(
            reconstructed_magnitudes=reconstructed_magnitudes,
        )
        return state_dot, out
