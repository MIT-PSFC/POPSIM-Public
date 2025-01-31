import os

import chex
import jax.numpy as jnp
import numpy as np
from interpax import Interpolator1D

from popsim import PACKAGE_ROOT, ModuleBase
from popsim.modules.tearing import Tearing

"""
Classes which simulate magnetic diagnostics
"""

"""
B Field Poloidal Probes. Just the magnetic field data from an arbitrary set of probes
"""


def measure_magnetic_field(
    probe_details: list[dict],
    filtered_mags: dict[tuple[int, int], float],
    mode_phases: dict[tuple[int, int], float],
    modes: list[tuple[int, int]],
) -> jnp.ndarray:
    """
    VERY PRELIMINARY IMPLEMENTATION TODO(ZanderKeith)
    Measurement of the magnetic field [T] at each probe, assuming cylindrical mode structure
    This is also sorta assuming the probes are zeroed out once the main background field gets ramped up.
    Also, should this be the T/s signal that you'd actually get from the probe or is there some integration weirdness going on?
    What is the thing that SPARC will be working with and that DEFUSE wants to see?
    """

    mode_perturbations = jnp.zeros((len(modes), len(probe_details)))

    for i, mode in enumerate(modes):
        mode_mag = filtered_mags[mode]
        mode_phase = mode_phases[mode]
        poloidal_mode_number = mode[0]
        toroidal_mode_number = mode[1]
        for j, probe in enumerate(probe_details):
            # Again, expecting something cylindrical-ish for right now. Also a little spaghetti because expects this to be added elsewhere.
            probe_theta = probe["position"]["theta"]
            probe_phi = probe["position"]["phi"]
            poloidal_phase_shift = (probe_theta - mode_phase) * poloidal_mode_number
            toroidal_phase_shift = (probe_phi - mode_phase) * toroidal_mode_number
            mode_perturbation = mode_mag * jnp.cos(poloidal_phase_shift + toroidal_phase_shift)
            mode_perturbations = mode_perturbations.at[i, j].set(mode_perturbation)

    total_perturbation = jnp.sum(mode_perturbations, axis=0)

    return total_perturbation


@chex.dataclass
class BFieldPoloidalProbes(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.

        # The transfer function from Bp to A for various frequencies
        func_Bp_per_A: Interpolator1D
        # List of dictionaries for all the probe details, as defined in the device description
        probe_details: list[dict]
        # Major radius. Idk where else to put this but it's needed for the cylindrical math
        R0: float

    @chex.dataclass
    class State:
        # Magnetics don't need to keep track of their own state
        pass

    @chex.dataclass
    class Output:
        # Measured magnetic field at each probe
        Bp: dict[str, float]  # T

    @chex.dataclass
    class Inputs:
        tearing_out: Tearing.Output  # The output of the Tearing module.
        modes: list[tuple[int, int]]

    config: Config

    def __init__(self, config):
        self.config = config

        # For each entry in the config, if there is no "theta" key under "position", calculate it and add it
        # Note that this is the angle from the midplane with the minor radius as the hypotenuse
        # Just pre-computing this to make the cylindrical and circular toroidal math more efficient later
        for i, probe in enumerate(self.config.probe_details):
            if "theta" not in probe["position"]:
                magnetic_axis_r = probe["position"]["r"] - self.config.R0
                probe["position"]["theta"] = jnp.arctan2(probe["position"]["z"], magnetic_axis_r)
                self.config.probe_details[i] = probe

    def __call__(self, state: State, inputs: Inputs) -> Output:
        """Measure the magnetic field at each probe in the B Field Poloidal Probes module.

        Args:
            state (State): The state of the B Field Poloidal Probes module. Not used in this module.
            inputs (Inputs): The parameters of the B Field Poloidal Probes module. Includes the output of the Tearing module and the tearing modes to reconstruct.

        Returns:
            Output: The output of the B Field Poloidal Probes module. Includes the measured magnetic field at each probe.
        """
        # Get the perturbed current, phase, and frequency of each tearing mode
        mode_currents = inputs.tearing_out.mode_current
        mode_phases = inputs.tearing_out.mode_phase
        mode_freqs = inputs.tearing_out.mode_freq

        # Convert to the signal that would be measured by the probes (adjusted by the Bp/A transfer function)
        filtered_signals = {mode: mode_currents[mode] * self.config.func_Bp_per_A(mode_freqs[mode]) for mode in inputs.modes}

        measured_signals_array = measure_magnetic_field(self.config.probe_details, filtered_signals, mode_phases, inputs.modes)

        # Turn measured signals into a dictionary
        measured_signals = {
            self.config.probe_details[i]["identifier"]: measured_signals_array[i] for i in range(len(measured_signals_array))
        }

        out = BFieldPoloidalProbes.Output(Bp=measured_signals)

        return out

    def default_setup(empty: bool = False):
        """Get a standard instance of the B Field Poloidal Probes module."""

        if empty:
            probe_details = []
        else:
            probe_details = [
                {
                    "area": 0.1,
                    "identifier": "sample_probe_1_identifier",
                    "name": "sample_probe_1_name",
                    "poloidal_angle": 0.0,
                    "position": {"phi": 0.0, "r": 1.0, "z": 0.0},
                    "type": {"index": 2},
                },
                {
                    "area": 0.1,
                    "identifier": "sample_probe_2_identifier",
                    "name": "sample_probe_2_name",
                    "poloidal_angle": 0.0,
                    "position": {"phi": 0.5, "r": 1.0, "z": 0.0},
                    "type": {"index": 2},
                },
            ]
        R0 = 1.0
        # TODO(ZanderKeith): again, should really be reading from the device description
        _, func_Bp_per_A = load_lown_config()

        b_field_poloidal_probes_config = BFieldPoloidalProbes.Config(func_Bp_per_A=func_Bp_per_A, probe_details=probe_details, R0=R0)

        b_field_poloidal_probes_module = BFieldPoloidalProbes(config=b_field_poloidal_probes_config)

        return b_field_poloidal_probes_module


"""
Low-N Array, for measuring the toroidal mode number of any type of magnetic perturbation using a differenced array of probes.
"""


def load_lown_config():
    """Load the configuration data for the Low-N array.
    TODO(ZanderKeith): This is a temporary function that will be replaced with a more general
    solution once we determine how to include device descriptions in the simulator.

    Returns:
        probe_connections: A list of tuples representing the connections between probes.
        func_Bp_per_A: The transfer function from Bp to A for various frequencies.
    """
    # Get sensor positions from lown_design.txt config file
    # The text file has one tuple of (phi1, phi2) per line
    lown_fullpath = os.path.join(PACKAGE_ROOT, "data/tearing/lown_design.txt")
    with open(lown_fullpath) as f:
        probe_connections = [tuple(map(float, line.split(", "))) for line in f]

    # Get frequency responses
    resp_fullpath = os.path.join(PACKAGE_ROOT, "data/tearing/21_mode_resp_data.txt")
    out = np.loadtxt(resp_fullpath, skiprows=1)

    freqs = out[:, 0]
    Bp_per_A = out[:, 1]  # Bp (poloidal field) per Amp of tearing mode current

    # If the maximum frequency is less than 100 kHz, extrapolate to higher frequencies and print a warning
    # TODO(ZanderKeith), should ask Ryan if we have higher frequency response data for magnetics
    max_freq = max(freqs)
    if max_freq < 100e3:
        print(f"Warning: Maximum frequency in response data is {max_freq / 1e3} kHz. Repeating response up to 100 kHz.")
        freqs = np.append(freqs, [100e3])
        Bp_per_A = np.append(Bp_per_A, [Bp_per_A[-1]])

    # Mirror the transfer function for negative frequencies
    freqs = np.concatenate((-freqs[::-1], freqs))
    Bp_per_A = np.concatenate((Bp_per_A[::-1], Bp_per_A))

    # Make Jax-compatible interpolators to determine
    # measured field per Amp of tearing mode current at arbitrary rotation frequencies
    func_Bp_per_A = Interpolator1D(freqs, Bp_per_A)

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
            differenced_signals = differenced_signals.at[i, j].set(
                probe_signal_2 - probe_signal_1
            )  # OMAS definition of differenced probe signal

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
        # Magnetics don't need to keep track of their own state
        pass

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        reconstructed_magnitudes: dict[int, float]  # T
        filtered_signals: dict[tuple[int, int], float]  # T
        differenced_signals: jnp.ndarray  # T

    @chex.dataclass
    class Inputs:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        tearing_out: Tearing.Output  # The output of the Tearing module.
        modes: list[
            tuple[int, int]
        ]  # The tearing modes in the tearing module. The modes that will be reconstructed are separate, and are set in the config.

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

    def __call__(self, state: State, inputs: Inputs) -> Output:
        """Reconstruct the magnitudes of the tearing modes from the signals measured by the probes in the Low-N array.

        Args:
            state (State): The state of the Low-N array module. Not used in this module.
            inputs (Inputs): The parameters of the Low-N array module. Includes the output of the Tearing module and the tearing modes to reconstruct.

        Returns:
            Output: The output of the Low-N array module. Includes the reconstructed magnitudes of the tearing modes, the filtered signals for each probe, and the differenced signals for each connection.
        """
        # Get the perturbed current, phase, and frequency of each tearing mode
        mode_currents = inputs.tearing_out.mode_current
        mode_phases = inputs.tearing_out.mode_phase
        mode_freqs = inputs.tearing_out.mode_freq

        # Convert to the signal that would be measured by the probes (adjusted by the Bp/A transfer function)
        filtered_signals = {mode: mode_currents[mode] * self.config.func_Bp_per_A(mode_freqs[mode]) for mode in inputs.modes}

        # Get the differenced signals between each pair of probes
        differenced_signals = get_differenced_signals(
            self.config.probe_connections,
            filtered_signals,
            mode_phases,
            inputs.modes,
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

    def default_setup(empty: bool = False):
        """Get a standard instance of the Low-N array module."""
        if empty:
            _, func_Bp_per_A = load_lown_config()
            probe_connections = []
        else:
            probe_connections, func_Bp_per_A = load_lown_config()

        lown_array_config = LowNArray.Config(
            func_Bp_per_A=func_Bp_per_A, probe_connections=probe_connections, reconstructed_modes=[1, 2, 3]
        )

        lown_array_module = LowNArray(config=lown_array_config)

        return lown_array_module
