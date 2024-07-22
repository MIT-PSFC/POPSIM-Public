import chex
import jax.numpy as jnp
import numpy as np
from interpax import Interpolator1D

from popsim import ModuleBase
from popsim.modules.tearing import Tearing

"""
Classes which simulate magnetic diagnostics
"""


def load_lown_config(filepath: str):
    # Get sensor positions from lown_design.txt config file
    fname = "lown_design.txt"
    # The text file has one tuple of (phi1, phi2) per line
    with open(filepath + fname) as f:
        probe_connections = [tuple(map(float, line.split(", "))) for line in f]

    # Get frequency responses
    fname = "21_mode_resp_data.txt"
    out = np.loadtxt(filepath + fname, skiprows=1)

    freq = out[:, 0]
    Bp_per_A = out[:, 1]  # Bp (poloidal field) per Amp of tearing mode current

    # Make Jax-compatible interpolators to determine
    # measured field per Amp of tearing mode current at arbitrary rotation frequencies
    func_Bp_per_A = Interpolator1D(freq, Bp_per_A)

    return probe_connections, func_Bp_per_A


def build_design_matrix(probe_connections: list[tuple[float, float]], measured_mode_numbers: jnp.ndarray) -> np.ndarray:
    """Build the design matrix that encodes the connections between probes and the mode numbers of the tearing modes being measured.
    This assumes that we are measuring modes n=1 up to the maximum mode number.

    Args:
        probe_connections (list[tuple[float, float]]): A list of tuples representing the connections between probes.
        measured_mode_numbers (jnp.ndarray): The mode numbers of the tearing modes being measured.

    Returns:
        jnp.ndarray: The design matrix.
    """

    # This is only done in the setup so we can use numpy
    design_matrix = np.zeros((len(probe_connections), 2 * len(measured_mode_numbers)))

    for i, (probe1_angle, probe2_angle) in enumerate(probe_connections):
        for j, mode_number in enumerate(measured_mode_numbers):
            design_matrix[i, 2 * j] = np.cos(mode_number * probe1_angle) - np.cos(mode_number * probe2_angle)
            design_matrix[i, 2 * j + 1] = np.sin(mode_number * probe1_angle) - np.sin(mode_number * probe2_angle)

    return design_matrix


def get_differenced_signals(
    probe_connections: list[tuple[float, float]],
    filtered_mags: dict[tuple[int, int], float],
    mode_phases: dict[tuple[int, int], float],
    modes: list[tuple[int, int]],
) -> jnp.ndarray:
    """The Low-N array has digitizers that record the differences between probes to isolate
    the small tearing mode signal (~0.001 T) on top of the large background field (~10 T)

    Args:
        probe_connections (list[tuple[float, float]]): A list of tuples representing the connections between probes.
        filtered_mags (dict[float, float]): The magnitudes of the tearing modes after being filtered by the transfer function.
        mode_phases: The phases of the tearing modes.
        modes: The tearing modes being measured.

    Returns:
        jnp.ndarray: The differenced signals.
    """

    differenced_signals = jnp.zeros((len(modes), len(probe_connections)))

    for i, mode in enumerate(modes):
        mode_mag = filtered_mags[mode]
        mode_phase = mode_phases[mode]
        toroidal_mode_number = mode[1]
        for j, (probe1_angle, probe2_angle) in enumerate(probe_connections):
            probe_signal_1 = mode_mag * jnp.cos(toroidal_mode_number * (probe1_angle - mode_phase))
            probe_signal_2 = mode_mag * jnp.cos(toroidal_mode_number * (probe2_angle - mode_phase))
            differenced_signals = differenced_signals.at[i, j].set(probe_signal_1 - probe_signal_2)

    # Sum the differenced signals for each mode
    differenced_signals = jnp.sum(differenced_signals, axis=0)

    return differenced_signals


@chex.dataclass
class LowNArray(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.

        # The transfer function from Bp to A for various frequencies
        func_Bp_per_A: Interpolator1D
        # The connections between the probes
        probe_connections: list[tuple[float, float]]
        # The mode numbers to reconstruct. Maximum should be len(probe_connections) / 2
        reconstructed_modes: list[int]

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        pass

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        reconstructed_magnitudes: dict[int, float]  # T
        filtered_signals: dict[tuple[int, int], float]  # T
        differenced_signals: jnp.ndarray  # T

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        pass

    config: Config
    pseudoinverse_matrix: jnp.ndarray
    measured_mode_numbers: jnp.ndarray

    def __init__(self, config):
        self.config = config

        # Make the matrix for reconstructing tearing mode magnitudes from probe measurements
        # Fill in mode numbers from 1 to the maximum
        # This appears to be a requirement of JAX, where the dictionary sorting does not necessarily
        # line up with the created arrays, so instead we just include all n=1 to n=max_n
        # and create the dictionary using the indices as keys
        self.measured_mode_numbers = jnp.arange(1, max(self.config.reconstructed_modes) + 1)
        design_matrix = build_design_matrix(self.config.probe_connections, self.measured_mode_numbers)
        # Needs to be jnp array for JAX
        self.pseudoinverse_matrix = jnp.linalg.pinv(design_matrix)

    def __call__(self, tearing_out: Tearing.Output, modes: list[tuple[int, int]]) -> Output:
        """Reconstruct the magnitudes of the tearing modes from the signals measured by the probes in the Low-N array.

        Args:
            tearing_out (Tearing.Output): The output of the Tearing module.
            modes (list[tuple[int, int]]): The tearing modes in the tearing module. The modes that will be reconstructed are separate, and are set in the config.

        Returns:
            Output: The output of the Low-N array module. Includes the reconstructed magnitudes of the tearing modes, the filtered signals for each probe, and the differenced signals for each connection.
        """
        # Get the perturbed current, phase, and frequency of each tearing mode
        mode_currents = tearing_out.mode_current
        mode_phases = tearing_out.mode_phase
        mode_freqs = tearing_out.mode_freq

        # Convert to the signal that would be measured by the probes (adjusted by the Bp/A transfer function)
        filtered_signals = {mode: mode_currents[mode] * self.config.func_Bp_per_A(mode_freqs[mode]) for mode in modes}

        # Get the differenced signals between each pair of probes
        differenced_signals = get_differenced_signals(
            self.config.probe_connections,
            filtered_signals,
            mode_phases,
            modes,
        )

        reconstructed_components = self.pseudoinverse_matrix @ differenced_signals

        # Reconstruct the magnitudes of the tearing modes
        reconstructed_magnitudes = {}
        for i in range(len(self.measured_mode_numbers)):
            reconstructed_magnitudes[i + 1] = jnp.sqrt(reconstructed_components[2 * i] ** 2 + reconstructed_components[2 * i + 1] ** 2)

        out = LowNArray.Output(
            reconstructed_magnitudes=reconstructed_magnitudes,
            filtered_signals=filtered_signals,
            differenced_signals=differenced_signals,
        )

        return out
