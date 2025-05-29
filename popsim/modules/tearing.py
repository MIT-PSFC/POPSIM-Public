import dataclasses
import json
from enum import IntEnum
from typing import Optional

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from popsim import PACKAGE_ROOT, TimeDepModuleBase, discrete_time_field
from popsim.logic_utils import select_w_tuples
from popsim.simulate import make_time_base

"""
Models the growth of tearing modes in a tokamak plasma.
"""


class DisruptionPhase(IntEnum):
    NONE = 0
    TQ = 1
    CQ = 2


class TearingPhase(IntEnum):
    NONE = 0
    SPAWN = 1
    ROTATING = 2
    DECELERATING = 3
    LOCKED = 4


class EFUniverse(IntEnum):
    """Choices of Monte-Carlo universe for shifts/tilts of active circuits.
    Either flattop or startup, 1st, 50th, and 99.9th percentile (higher means larger EF overlap).
    """

    STARTUP_01 = 3
    STARTUP_50 = 4
    STARTUP_99p9 = 5
    FLATTOP_01 = 0
    FLATTOP_50 = 1
    FLATTOP_99p9 = 2


class OverlapPhase(IntEnum):
    STARTUP = 0
    FLATTOP = 1


DEFAULT_WDOT = {
    (3, 2): 4e-2 / 0.5,  # m/s
    (2, 1): 10e-2 / 0.5,
    (3, 1): 3e-2 / 0.5,
}


TQ_WDOT = {
    (3, 2): 4e-2 / 0.002,  # m/s
    (2, 1): 10e-2 / 0.002,
    (3, 1): 3e-2 / 0.002,
}


CQ_WDOT = {
    (3, 2): -4e-2 / 0.005,  # m/s
    (2, 1): -10e-2 / 0.005,
    (3, 1): -3e-2 / 0.005,
}


INITIAL_ROT_FREQ = {
    (3, 2): 7e3 * 1.5,  # Hz
    (2, 1): 7e3,
    (3, 1): 7e3,
}


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def generate_disruption_phase_trajectory(
    trigger_time: float, tq_to_cq_dur: float, time_base: np.ndarray, dt: Optional[float] = 1e-4 / 3
) -> dict[float, DisruptionPhase]:
    # TODO(allenw): we want a rectilinear interpolation scheme.
    disrupt_phase_dict = {
        0.0: DisruptionPhase.NONE,
        trigger_time - dt: DisruptionPhase.NONE,
        trigger_time: DisruptionPhase.TQ,
        trigger_time + tq_to_cq_dur - dt: DisruptionPhase.TQ,
        trigger_time + tq_to_cq_dur: DisruptionPhase.CQ,
    }

    # Round the times to the nearest time step in the time base.
    disrupt_phase_dict = {find_nearest(time_base, time): phase for time, phase in disrupt_phase_dict.items()}
    return disrupt_phase_dict


def generate_tearing_phase_trajectory(
    trigger_time: float, rot_dur: float, locking_dur: float, time_base: np.ndarray, dt: Optional[float] = 1e-4 / 3
):
    # TODO(allenw): we want a rectilinear interpolation scheme.
    rot_phase_dict = {
        0.0: TearingPhase.NONE,
        trigger_time - dt: TearingPhase.NONE,
        trigger_time: TearingPhase.SPAWN,
        trigger_time + dt: TearingPhase.ROTATING,
        trigger_time + rot_dur - dt: TearingPhase.ROTATING,
        trigger_time + rot_dur: TearingPhase.DECELERATING,
        trigger_time + rot_dur + locking_dur - dt: TearingPhase.DECELERATING,
        trigger_time + rot_dur + locking_dur: TearingPhase.LOCKED,
    }

    # Round the times to the nearest time step in the time base.
    rot_phase_dict = {find_nearest(time_base, time): phase for time, phase in rot_phase_dict.items()}
    return rot_phase_dict


def calculate_tearing_growth_rate(
    disruption_phase: DisruptionPhase, rotation_phase: TearingPhase, default_Wdot: float, TQ_Wdot: float, CQ_Wdot: float
) -> ArrayLike:
    """Calculate the tearing mode's growth rate based on its rotation phase and the disruption phase.
    This presently returns hard-coded values for the growth rate, but in the future
    this could be replaced with a more sophisticated model.

    Args:
        disruption_phase (DisruptionPhase): The phase of the disruption.
        rotation_phase (TearingPhase): The phase of the tearing mode's rotation.
        default_Wdot (float): The default growth rate of the tearing mode.


    Returns:
        ArrayLike: The growth rate of the tearing mode.
    """

    conditions_to_choices = [
        (rotation_phase == TearingPhase.NONE, 0.0),
        (disruption_phase == DisruptionPhase.NONE, default_Wdot),
        (disruption_phase == DisruptionPhase.TQ, TQ_Wdot),
        (disruption_phase == DisruptionPhase.CQ, CQ_Wdot),
    ]

    Wdot = select_w_tuples(conditions_to_choices, default=default_Wdot)
    return Wdot


def calculate_rotation_dot(tearing_phase: TearingPhase, rot_dur: float, locking_dur: float, initial_rot_freq: float) -> ArrayLike:
    """Calculate the time derivative of the rotation frequency based on the rotation phase.

    Args:
        tearing_phase (TearingPhase): The phase of the tearing mode's evolution.
        rot_dur (float): How long the mode rotates before beginning deceleration.
        locking_dur (float): How long the mode decelerates before locking.
        initial_rot_freq (float): The initial rotation frequency of the mode.

    Returns:
        ArrayLike: The time derivative of the mode's rotation frequency.
    """

    conditions_and_choices = [
        (tearing_phase == TearingPhase.NONE, 0.0),
        (tearing_phase == TearingPhase.ROTATING, -(initial_rot_freq / 2.0) / (rot_dur)),
        (tearing_phase == TearingPhase.DECELERATING, -(initial_rot_freq / 2.0) / (locking_dur)),
        (tearing_phase == TearingPhase.LOCKED, 0.0),
    ]

    Fdot = select_w_tuples(conditions_and_choices, default=0.0)
    return Fdot


class Tearing(TimeDepModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.
        modes: list[tuple[int, int]]

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        W: dict[tuple[int, int], float]  # [m], Nxm
        F: dict[tuple[int, int], float]  # [Hz]
        mode_phase: dict[tuple[int, int], float]  # [rad]

    @chex.dataclass
    class Inputs:
        # Define the, possibly time dependent, inputs that will be passed to the module.
        rot_dur: float  # Time from trigger to when the mode begins to slow down [s]
        locking_dur: float  # Time from when the mode begins to slow down to when it locks [s]
        disruption_phase: DisruptionPhase
        tearing_phase: TearingPhase
        cur_per_W: float = 1e3 / 1e-2  # Perturbed current per island width [A/m] TODO(ZanderKeith) a guess for now
        default_wdot: dict[tuple, float] = dataclasses.field(default_factory=lambda: DEFAULT_WDOT)
        tq_wdot: dict[tuple, float] = dataclasses.field(default_factory=lambda: TQ_WDOT)
        cq_wdot: dict[tuple, float] = dataclasses.field(default_factory=lambda: CQ_WDOT)
        initial_rot_freq: dict[tuple, float] = dataclasses.field(default_factory=lambda: INITIAL_ROT_FREQ)

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        inputs: "Inputs"  # noqa: F821
        mode_current: dict[tuple[int, int], float]  # A
        mode_phase: dict[tuple[int, int], float]  # rad
        mode_freq: dict[tuple[int, int], float]  # Hz
        aux_data: dict = None

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        disruption_phase = round(inputs.disruption_phase)
        tearing_phase = round(inputs.tearing_phase)

        Wdot = {
            mode: calculate_tearing_growth_rate(
                disruption_phase,
                tearing_phase,
                inputs.default_wdot[mode[0], mode[1]],
                inputs.tq_wdot[mode[0], mode[1]],
                inputs.cq_wdot[mode[0], mode[1]],
            )
            for mode in self.config.modes
        }
        Fdot = {
            mode: calculate_rotation_dot(tearing_phase, inputs.rot_dur, inputs.locking_dur, inputs.initial_rot_freq[mode[0], mode[1]])
            for mode in self.config.modes
        }
        mode_phase_dot = {mode: state.F[mode] * 2 * jnp.pi for mode in self.config.modes}

        for mode in self.config.modes:
            # Force W to stay positive
            state.W[mode] = jnp.where(state.W[mode] > 0, state.W[mode], 0.0)

            # If mode just spawned, set F to initial value
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.SPAWN, inputs.initial_rot_freq[mode[0], mode[1]], state.F[mode])
            # If mode is locked, set F to 0
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.LOCKED, 0.0, state.F[mode])

        # Calculate perturbed current
        perturbed_current = {mode: state.W[mode] * inputs.cur_per_W for mode in self.config.modes}

        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot)

        # Make an output.
        out = Tearing.Output(
            state_dot=state_dot,
            inputs=inputs,
            mode_current=perturbed_current,
            mode_phase=state.mode_phase,
            mode_freq=state.F,
        )
        return state_dot, out

    def default_setup(empty: bool = False):
        """Get a standard instance, initial state, and inputs for the Tearing module.

        Args:
        -----
        empty : bool, optional (default=False)
            If True, the modes will be empty.

        Returns:
        --------
        tearing_module : Tearing

        tearing_initial_state : Tearing.State

        tearing_inputs : Tearing.Inputs

        time_base : np.ndarray
        """

        if empty:
            modes = []
        else:
            modes = [(2, 1), (3, 1)]
        tearing_config = Tearing.Config(
            modes=modes,
        )

        tearing_initial_state = Tearing.State(
            W={mode: 0.0 for mode in modes}, F={mode: 0.0 for mode in modes}, mode_phase={mode: 0.0 for mode in modes}
        )

        dt = 1e-4 / 3  # s
        time_base = make_time_base(t0=0.0, t1=4.0, dt=dt)

        rot_dur = 1.0
        locking_dur = 0.2
        trigger_time = 2.0
        disrupt_time = 3.5
        dur_tq_to_spike = 1e-3

        tearing_inputs = Tearing.Inputs(
            rot_dur=rot_dur,  # s
            locking_dur=locking_dur,  # s
            disruption_phase=generate_disruption_phase_trajectory(disrupt_time, dur_tq_to_spike, time_base, dt),
            tearing_phase=generate_tearing_phase_trajectory(trigger_time, rot_dur, locking_dur, time_base, dt),
        )

        tearing_module = Tearing(config=tearing_config)

        return tearing_module, tearing_initial_state, tearing_inputs, time_base


def load_active_circuit_overlaps(
    overlap_phase: OverlapPhase, ef_universe: EFUniverse
) -> tuple[dict[str, dict[str, complex]], dict[str, list[str]]]:
    """
    Load the overlap data for the active circuits. This is mostly just a placeholder until we get the real data.

    Args:
        overlap_phase (OverlapPhase): The phase of the overlap data to load.
        ef_universe (EFUniverse): The universe of the EF data to load.

    Returns:
        overlaps (dict[str, dict[str, complex]]): The overlap data for each coil source in terms of delta per amp.
    """

    rename_dict = {
        "divl": ["div1l", "div2l"],
        "divu": ["div1u", "div2u"],
    }

    # Load overlaps for active circuits
    # Nominal overlaps are given in delta per amp, while shift and tilt are given in delta per m displacement.
    if overlap_phase == OverlapPhase.STARTUP:
        error_field_source_file = f"{PACKAGE_ROOT}/data/tearing/error_field_sources/startup_01132025.json"
    elif overlap_phase == OverlapPhase.FLATTOP:
        error_field_source_file = f"{PACKAGE_ROOT}/data/tearing/error_field_sources/flattop_01132025.json"
    else:
        raise ValueError(f"Invalid overlap_phase: {overlap_phase}")

    with open(error_field_source_file) as f:
        overlap_data = json.load(f)

    # Load perturbation shifts/tilts for active circuits
    if ef_universe in [EFUniverse.STARTUP_01, EFUniverse.STARTUP_50, EFUniverse.STARTUP_99p9]:
        perts_file = f"{PACKAGE_ROOT}/data/tearing/error_field_sources/perts_startup.json"
    elif ef_universe in [EFUniverse.FLATTOP_01, EFUniverse.FLATTOP_50, EFUniverse.FLATTOP_99p9]:
        perts_file = f"{PACKAGE_ROOT}/data/tearing/error_field_sources/perts_flattop.json"

    with open(perts_file) as f:
        perts_data = json.load(f)

    if ef_universe in [EFUniverse.STARTUP_01, EFUniverse.FLATTOP_01]:
        perts = perts_data["1st_percentile"]
    elif ef_universe in [EFUniverse.STARTUP_50, EFUniverse.FLATTOP_50]:
        perts = perts_data["50th_percentile"]
    elif ef_universe in [EFUniverse.STARTUP_99p9, EFUniverse.FLATTOP_99p9]:
        perts = perts_data["99p9th_percentile"]

    overlaps = {}

    # List of name tuples, where the first is the actual name and the second is the name in the error field source file
    name_pairs = [
        (coil_name, data_key) if data_key in rename_dict else (data_key, data_key)
        for data_key in overlap_data
        for coil_name in rename_dict.get(data_key, [data_key])
    ]

    np.random.seed(0)  # Reproducibility of the random shift/tilt and phase between startup and flattop for now until we get the real data
    for name_pair in name_pairs:
        coil_name = name_pair[0]
        data_key = name_pair[1]
        data = overlap_data[data_key]

        overlaps_single = {}

        # Nominal is in terms of delta per amp, so this is easy (will be multiplied by actual current later)
        if "nominal" in data:
            overlaps_single[coil_name] = complex(data["nominal"])

        # Shift and tilt are in terms of delta per amp per meter displacement (or perturbation)
        for source in ["shift", "tilt"]:
            base_factor = complex(data[source])
            pert = complex(perts[data_key][source])

            # Convert into delta per amp
            source_overlap = base_factor * pert

            overlaps_single[coil_name] = overlaps_single.get(coil_name, 0) + source_overlap

        for key, value in overlaps_single.items():
            overlaps[key] = value

    return overlaps


def load_tf_overlap(overlap_percentile: EFUniverse | float) -> dict[str, complex]:
    """
    Load the cumulative overlap data for all TF coils.

    Args:
        overlap_percentile: The percentile of the probable overlap distribution to use for the TF coils. Either a float (0-1) or a EFUniverse enum.

    Returns:
        tf_overlap (dict[str, complex]): The overlap data for the TF coils.
    """

    if isinstance(overlap_percentile, EFUniverse):
        if overlap_percentile in [EFUniverse.STARTUP_01, EFUniverse.FLATTOP_01]:
            overlap_percentile = 0.01
        elif overlap_percentile in [EFUniverse.STARTUP_50, EFUniverse.FLATTOP_50]:
            overlap_percentile = 0.5
        elif overlap_percentile in [EFUniverse.STARTUP_99p9, EFUniverse.FLATTOP_99p9]:
            overlap_percentile = 0.999
        else:
            raise ValueError(f"Invalid EFUniverse: {overlap_percentile}")

    elif isinstance(overlap_percentile, float):
        if overlap_percentile < 0 or overlap_percentile > 1:
            raise ValueError(f"overlap_percentile must be between 0 and 1, got {overlap_percentile}")
    else:
        raise ValueError(f"overlap_percentile must be a float or EFUniverse enum, got {type(overlap_percentile)}")

    tf_overlap_file = f"{PACKAGE_ROOT}/data/tearing/error_field_sources/tfef.dat"
    data = np.genfromtxt(tf_overlap_file, dtype=float, delimiter="  ").T

    overlaps = data[0]
    percentiles = data[1]

    # Interpolate the data to get the error field at the desired percentile
    tf_overlap = np.interp(overlap_percentile, percentiles, overlaps)

    return tf_overlap


def calculate_total_overlap(
    config: "ErrorFieldLocking.Config",
    delta_static: complex,
    active_circuit_currents: dict[str, float],
    active_circuit_overlaps: dict[str, complex],
) -> complex:
    """
    Calculates the total overlap of the EF sources based on the
    simulation's current state.
    """

    # Instantaneous overlap starts with the TF's and static sources
    overlap_inst = config.tf_overlap + delta_static

    for circuit, current in active_circuit_currents.items():
        overlap_inst += current * active_circuit_overlaps[circuit]

    overlap_inst *= config.efc_efficiency

    # Only get the magnitude at the end. Up to here, the overlap is a complex number to allow constructive/destructive interference.
    return jnp.abs(overlap_inst)


def calculate_locking_threshold(scaling_law_inputs: dict[str, float], scaling_law_terms: dict[str, list[float]]) -> float:
    """
    Calculate the locking threshold based on an arbitrary scaling law.

    Args:
        scaling_law_inputs (dict[str, float]): The input values of each parameter in the scaling law.
        scaling_law_terms (dict[str, list[float]]): The terms for the scaling law, where the first element is the power of the parameter and the second is the error.

    Returns:
        float: The locking threshold.
    """

    if scaling_law_inputs.keys() != scaling_law_terms.keys():
        raise ValueError("scaling_law_inputs and scaling_law_terms must have the same keys")

    delta = 1.0
    for key, value in scaling_law_inputs.items():
        delta *= value ** scaling_law_terms[key][0]

    return delta


def locked_mode_dynamics(
    config: "ErrorFieldLocking.Config",
    state: "ErrorFieldLocking.State",
    overlap: float,
    locking_threshold: float,
    rational_surface_exists: bool,
) -> TearingPhase:
    """Determine how the locked mode evolves based on the overlap and the locking threshold.

    If there is no mode, and the overlap exceeds the locking threshold, then a locked mode will form.
    If there is a locked mode, and the overlap drops below the hystereis fraction of the locking threshold, then the mode will go away.
    If there is no rational surface, then the mode is forced unlocked.

    Args:
        config (ErrorFieldLocking.Config): The configuration for the module.
        state (ErrorFieldLocking.State): The state of the module.
        overlap (float): The present overlap of the error field.
        locking_threshold (float): The presently calculated empirical threshold of a locked mode forming

    Returns:
        TearingPhase: The new phase of the mode
    """

    # TODO(zkeith): In the future when we may need to do more than just locked and unlocked, we should make this more sophisticated.
    # Might get a little screwy since select_w_tuples only works with one condition at a time.
    threshold_check = [
        (rational_surface_exists == 0, TearingPhase.NONE),
        (overlap > locking_threshold, TearingPhase.LOCKED),
        (overlap < locking_threshold * config.hysteresis, TearingPhase.NONE),
    ]

    new_tearing_phase = select_w_tuples(threshold_check, default=state.tearing_phase)

    return new_tearing_phase


class ErrorFieldLocking(TimeDepModuleBase):
    @chex.dataclass
    class Config:
        # Combined overlap from the TF coils (pre-calculated since this shouldn't change over time)
        tf_overlap: float

        # Static EF sources from ferritic materials in the Tokamak Hall.
        static_source_overlaps: dict[str, complex]

        # The phase offset of the first TF coil proceeding from the +x axis counter-clockwise
        tf_phase_offset: float = 0.0

        # The modes that are being considered for locking. 2/1 is most dangerous so that's the one we're doing for now.
        modes: list[tuple[int, int]] = dataclasses.field(default_factory=lambda: [(2, 1)])

        # Hysteresis fraction of the locked mode.
        # This is the fraction of the threshold that must be reached to unlock the mode.
        # E.g, 0.9 means a point below 90% of the threshold must be reached to unlock.
        hysteresis: float = 0.9

        # How much of the error field is reduced by the correction coils.
        # There is precedent for this naive modeling approach, can be made more sophisticated later.
        efc_efficiency: float = 0.5

    @chex.dataclass
    class State:
        tearing_phase: TearingPhase = discrete_time_field()
        W: dict[tuple[int, int], float] = dataclasses.field(default_factory=lambda: {mode: 0.0 for mode in [(2, 1)]})
        F: dict[tuple[int, int], float] = dataclasses.field(default_factory=lambda: {mode: 0.0 for mode in [(2, 1)]})
        mode_phase: dict[tuple[int, int], float] = dataclasses.field(default_factory=lambda: {mode: 0.0 for mode in [(2, 1)]})

    @chex.dataclass
    class Inputs:
        scaling_law_terms: dict[str, list[float]]  # Terms in the scaling law, see data/tearing/scalinglaws.json for examples
        scaling_law_inputs: dict[str, float]  # Values for each parameter in the scaling law
        active_circuit_currents: dict[str, float]  # Current in active circuits [A]
        active_circuit_overlaps: dict[str, dict[str, complex]]  # Overlaps for each coil source [delta per A]
        rational_surface_exists: int  # Placeholder. Must have some term (be it q90 or something) that indicates the existence of a rational surface
        cur_per_W: float = 1e3 / 1e-2  # Perturbed current per island width [A/m] TODO(ZanderKeith) a guess for now

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        inputs: "Inputs"  # noqa: F821
        mode_current: dict[tuple[int, int], float]
        mode_phase: dict[tuple[int, int], float]
        mode_freq: dict[tuple[int, int], float]
        total_overlap: dict[tuple[int, int], float]
        locking_threshold: dict[tuple[int, int], float]
        aux_data: dict = None

    config: Config
    static_overlap: complex

    def __init__(self, config):
        self.config = config
        self.static_overlap = sum(config.static_source_overlaps.values())

    def __call__(
        self, state: "ErrorFieldLocking.State", inputs: "ErrorFieldLocking.Inputs"
    ) -> tuple["ErrorFieldLocking.State", "ErrorFieldLocking.Output"]:
        # Just doing the transition from none -> locked -> none for now

        total_overlap = calculate_total_overlap(
            self.config, self.static_overlap, inputs.active_circuit_currents, inputs.active_circuit_overlaps
        )
        locking_threshold = calculate_locking_threshold(inputs["scaling_law_inputs"], inputs["scaling_law_terms"])

        # If the overlap is greater than the threshold, the mode is locked
        new_tearing_phase = locked_mode_dynamics(self.config, state, total_overlap, locking_threshold, inputs.rational_surface_exists)

        for mode in self.config.modes:
            state.W[mode] = jnp.where(new_tearing_phase == TearingPhase.LOCKED, 1e-2, 0)

        Wdot = {mode: 0.0 for mode in self.config.modes}
        Fdot = {mode: 0.0 for mode in self.config.modes}
        mode_phase_dot = {mode: 0.0 for mode in self.config.modes}

        state_dot = ErrorFieldLocking.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot, tearing_phase=new_tearing_phase)

        out = ErrorFieldLocking.Output(
            state_dot=state_dot,
            inputs=inputs,
            mode_current={},
            mode_phase={},
            mode_freq={},
            total_overlap=total_overlap,
            locking_threshold=locking_threshold,
        )

        return state_dot, out
