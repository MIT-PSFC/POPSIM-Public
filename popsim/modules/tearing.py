import dataclasses
import json
from enum import IntEnum
from typing import Optional

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from popsim import ModuleBase, discrete_time_field
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


@chex.dataclass
class Tearing(ModuleBase):
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
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
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
        params: "Params"  # noqa: F821
        mode_current: dict[tuple[int, int], float]  # A
        mode_phase: dict[tuple[int, int], float]  # rad
        mode_freq: dict[tuple[int, int], float]  # Hz
        aux_data: dict = None

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        disruption_phase = round(params.disruption_phase)
        tearing_phase = round(params.tearing_phase)

        Wdot = {
            mode: calculate_tearing_growth_rate(
                disruption_phase,
                tearing_phase,
                params.default_wdot[mode[0], mode[1]],
                params.tq_wdot[mode[0], mode[1]],
                params.cq_wdot[mode[0], mode[1]],
            )
            for mode in self.config.modes
        }
        Fdot = {
            mode: calculate_rotation_dot(tearing_phase, params.rot_dur, params.locking_dur, params.initial_rot_freq[mode[0], mode[1]])
            for mode in self.config.modes
        }
        mode_phase_dot = {mode: state.F[mode] * 2 * jnp.pi for mode in self.config.modes}

        for mode in self.config.modes:
            # Force W to stay positive
            state.W[mode] = jnp.where(state.W[mode] > 0, state.W[mode], 0.0)

            # If mode just spawned, set F to initial value
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.SPAWN, params.initial_rot_freq[mode[0], mode[1]], state.F[mode])
            # If mode is locked, set F to 0
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.LOCKED, 0.0, state.F[mode])

        # Calculate perturbed current
        perturbed_current = {mode: state.W[mode] * params.cur_per_W for mode in self.config.modes}

        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot)

        # Make an output.
        out = Tearing.Output(
            state_dot=state_dot,
            params=params,
            mode_current=perturbed_current,
            mode_phase=state.mode_phase,
            mode_freq=state.F,
        )
        return state_dot, out

    def default_setup(empty: bool = False):
        """Get a standard instance, initial state, and params for the Tearing module.

        Args:
        -----
        empty : bool, optional (default=False)
            If True, the modes will be empty.

        Returns:
        --------
        tearing_module : Tearing

        tearing_initial_state : Tearing.State

        tearing_params : Tearing.Params

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

        tearing_params = Tearing.Params(
            rot_dur=rot_dur,  # s
            locking_dur=locking_dur,  # s
            disruption_phase=generate_disruption_phase_trajectory(disrupt_time, dur_tq_to_spike, time_base, dt),
            tearing_phase=generate_tearing_phase_trajectory(trigger_time, rot_dur, locking_dur, time_base, dt),
        )

        tearing_module = Tearing(config=tearing_config)

        return tearing_module, tearing_initial_state, tearing_params, time_base


def load_overlaps_and_sources(error_field_source_file: str) -> tuple[dict[str, dict[str, complex]], dict[str, list[str]]]:
    """
    Load the overlap data and sources for the error field locking module.

    Args:
        error_field_source_file (str): The path to the JSON file containing the overlap data and sources. Overlaps are given in delta per amp
        tf_error_field_file (str): The path to a JSON file containing the error field data for the TF coils.
        tf_error_field_percentile (float): The percentile of the error field contribution to use for the TF coils

    Returns:
        overlaps (dict[str, dict[str, complex]]): The overlap data for each coil source.
        coil_sources (dict[str, list[str]]): The sources for each coil.
    """

    rename_dict = {
        "divl": ["div1l", "div2l"],
        "divu": ["div1u", "div2u"],
    }

    with open(error_field_source_file) as f:
        overlap_data = json.load(f)

    overlaps = {}
    coil_sources = {}

    # Get overlaps and sources for non-TF coils
    for data_coil, data in overlap_data.items():
        overlaps_single = {}
        coil_sources_single = []

        for source in ["nominal"]:
            if source in data:
                overlaps_single[source] = data[source] * (1.0 + 0.00000001j)
                coil_sources_single.append(source)

        # Special cases for minor renaming
        if data_coil in rename_dict:
            for renamed_coil in rename_dict[data_coil]:
                overlaps[renamed_coil] = overlaps_single
                coil_sources[renamed_coil] = coil_sources_single
        else:
            overlaps[data_coil] = overlaps_single
            coil_sources[data_coil] = coil_sources_single

    return overlaps, coil_sources


def load_tf_overlap(tf_overlap_file: str, overlap_percentile: float) -> dict[str, complex]:
    """
    Load the cumulative overlap data for all TF coils.

    Args:
        tf_overlap_file (str): The path to a .dat file containing the overlap data for the TF coils.
        overlap_percentile (float): The percentile of the probable overlap distribution to use for the TF coils (0-1).

    Returns:
        tf_overlap (dict[str, complex]): The overlap data for the TF coils.
    """

    if overlap_percentile < 0 or overlap_percentile > 1:
        raise ValueError("overlap_percentile must be between 0 and 1")

    data = np.genfromtxt(tf_overlap_file, dtype=float, delimiter="  ").T

    overlaps = data[0]
    percentiles = data[1]

    # Interpolate the data to get the error field at the desired percentile
    tf_overlap = np.interp(overlap_percentile, percentiles, overlaps)

    return tf_overlap


def calculate_error_field_overlap(
    config: "ErrorFieldLocking.Config",
    delta_static: complex,
    pf_active_circuit_current: dict[str, float],
    overlaps: dict[str, dict[str, complex]],
) -> complex:
    """
    Calculates the total overlap of the EF sources based on the
    simulation's current state.
    """

    # Instantaneous overlap starts with that from the TF's and static sources
    overlap_inst = config.tf_overlap + delta_static

    # for coil in pfcscoils:
    for coil, current in pf_active_circuit_current.items():
        for source in config.coil_sources[coil]:
            overlap_inst += current * overlaps[coil][source]

    return overlap_inst


def calculate_locking_threshold(scaling_law_params, scaling_law_terms):
    """
    Calculate the locking scaling law.

    NOTE: This does not work well when the simulation includes early time points at which ne=0. This will cause the scaling to be 0.
    Thus, we have added a catch that checks that ne > 0.3e20 m^-3. This needs to be addressed before integrating with POPSIM.

    TODO(ZanderKeith): Put everything in SI units, and make sure the scaling law is correct.
    """

    # We're assuming that the parameters are a superset of the terms
    delta = 1.0
    for key, value in scaling_law_params.items():
        if key in scaling_law_terms:
            if key == "ne":
                delta *= jnp.maximum(value, 3) ** scaling_law_terms[key][0]
            else:
                delta *= value ** scaling_law_terms[key][0]

    return delta


def locked_mode_dynamics(
    config: "ErrorFieldLocking.Config", state: "ErrorFieldLocking.State", overlap: float, locking_threshold: float
) -> TearingPhase:
    """Determine how the locked mode evolves based on the overlap and the locking threshold.

    If there is no mode, and the overlap exceeds the locking threshold, then a locked mode will form.
    If there is a locked mode, and the overlap drops below the hystereis fraction of the locking threshold, then the mode will go away.

    Args:
        config (ErrorFieldLocking.Config): The configuration for the module.
        state (ErrorFieldLocking.State): The state of the module.
        overlap (float): The overlap of the error field.
        locking_threshold (float): The threshold for locking the mode.

    Returns:
        TearingPhase: The new phase of the mode
    """

    # TODO(zkeith): In the future when we may need to do more than just locked and unlocked, we should make this more sophisticated.
    # Might get a little screwy since select_w_tuples only works with one condition at a time.
    threshold_check = [
        (overlap > locking_threshold, TearingPhase.LOCKED),
        (overlap < locking_threshold * config.hysteresis, TearingPhase.NONE),
    ]

    new_tearing_phase = select_w_tuples(threshold_check, default=state.tearing_phase)

    return new_tearing_phase


@chex.dataclass
class ErrorFieldLocking(ModuleBase):
    @chex.dataclass
    class Config:
        # Total overlap from the TF coils (pre-calculated since this shouldn't change over time)
        tf_overlap: float

        # Static EF sources from ferritic materials in the Tokamak Hall.
        static_sources: dict[str, complex]

        # EF sources from PF/CS/any coil with a current that changes over time, in terms of delta per amp
        coil_sources: dict[str, list[str]]

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
    class Params:
        scaling_law_terms: dict[str, list[float]]  # Terms in the scaling law, see data/tearing/scalinglaws.json for examples
        scaling_law_params: dict[str, float]  # Values for each parameter in the scaling law
        pf_active_circuit_current: dict[str, float]  # Current in PF coils over time
        overlaps: dict[str, dict[str, complex]]  # Overlaps for each coil source
        cur_per_W: float = 1e3 / 1e-2  # Perturbed current per island width [A/m] TODO(ZanderKeith) a guess for now

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        params: "Params"  # noqa: F821
        mode_current: dict[tuple[int, int], float]
        mode_phase: dict[tuple[int, int], float]
        mode_freq: dict[tuple[int, int], float]
        error_field_overlap: dict[tuple[int, int], float]
        locking_threshold: dict[tuple[int, int], float]
        aux_data: dict = None

    config: Config
    delta_static: complex

    def __init__(self, config):
        self.config = config

        # TODO(zkeith): verify with Matt that this calculation is no longer necessary with new data file
        # Also ask what current was in the TF's for this data file, and how much is expected to change between 12T and 8T operation phase = np.exp(1j * 2 * np.pi * i / n_tf + config.tf_phase_offset)

        self.delta_static: complex = np.sum([self.config.overlaps[key]["nominal"] for key in self.config.static_sources])

    def __call__(
        self, state: "ErrorFieldLocking.State", params: "ErrorFieldLocking.Params"
    ) -> tuple["ErrorFieldLocking.State", "ErrorFieldLocking.Output"]:
        # Just doing the transition from none -> locked -> none for now

        error_field_overlap = calculate_error_field_overlap(
            self.config, self.delta_static, params.pf_active_circuit_current, params.overlaps
        )
        locking_threshold = calculate_locking_threshold(params["scaling_law_params"], params["scaling_law_terms"])

        # If the overlap is greater than the threshold, the mode is locked
        new_tearing_phase = locked_mode_dynamics(self.config, state, error_field_overlap, locking_threshold)

        # TODO (zkeith): Right now we're only doing the 2/1 mode, I'm unsure if we need to account for all modes in this.
        for mode in self.config.modes:
            state.W[mode] = jnp.where(new_tearing_phase == TearingPhase.LOCKED, 1e-2, 0)

        Wdot = {mode: 0.0 for mode in self.config.modes}
        Fdot = {mode: 0.0 for mode in self.config.modes}
        mode_phase_dot = {mode: 0.0 for mode in self.config.modes}

        state_dot = ErrorFieldLocking.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot, tearing_phase=new_tearing_phase)

        out = ErrorFieldLocking.Output(
            state_dot=state_dot,
            params=params,
            mode_current={},
            mode_phase={},
            mode_freq={},
            error_field_overlap=error_field_overlap,
            locking_threshold=locking_threshold,
        )

        return state_dot, out
