from enum import IntEnum

import chex
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from jax import lax

from popsim import ModuleBase
from popsim.logic_utils import select_w_tuples

"""
An example template to copy and paste when creating a new module.
"""


class DisruptionPhase(IntEnum):
    NONE = 0
    TQ = 1
    CQ = 2


class IslandRotationPhase(IntEnum):
    NONE = 0
    SPAWN = 1 # Need to think about how we want to implement the initial 'kick' that starts and island
    ROTATING = 2
    DECELERATING = 3
    LOCKED = 4

class IslandModeNumber(IntEnum):
    # Mode number is n/m, toroidal turns / poloidal turns 
    THREE_TWO = 0
    TWO_ONE = 1
    THREE_ONE = 2

# Hard-coded values for the island growth rates
default_Wdot_dict = {
    IslandModeNumber.THREE_TWO: 4e-2/0.5,  # m/s
    IslandModeNumber.TWO_ONE: 10e-2/0.5,
    IslandModeNumber.THREE_ONE: 3e-2/0.5
}
TQ_Wdot_dict = {
    IslandModeNumber.THREE_TWO: 4e-2/0.002,  # m/s
    IslandModeNumber.TWO_ONE: 10e-2/0.002,
    IslandModeNumber.THREE_ONE: 3e-2/0.002
}
CQ_Wdot_dict = {
    IslandModeNumber.THREE_TWO: -4e-2/0.005,  # m/s
    IslandModeNumber.TWO_ONE: -10e-2/0.005,
    IslandModeNumber.THREE_ONE: -3e-2/0.005
}

# Hard-coded values for the initial rotation frequency of the island
initial_rot_freq_dict = {
    IslandModeNumber.THREE_TWO: 7e3*1.5,
    IslandModeNumber.TWO_ONE: 7e3,
    IslandModeNumber.THREE_ONE: 7e3*0.67,
}

def calculate_tearing_growth_rate(disruption_phase: DisruptionPhase, rotation_phase: IslandRotationPhase, mode_number: IslandModeNumber) -> ArrayLike:
    """Calculate the tearing growth rate based on its rotation phase and the disruption phase.
    This currently returns hard-coded values for the growth rate, but in the future
    this could be replaced with a more sophisticated model.

    Args:
        rotation_phase (IslandRotationPhase): The phase of the island rotation.

    Returns:
        ArrayLike: The growth rate of the tearing mode.
    """

    default_Wdot = default_Wdot_dict[mode_number]
    TQ_Wdot = TQ_Wdot_dict[mode_number]
    CQ_Wdot = CQ_Wdot_dict[mode_number]

    conditions_to_choices = [
        (rotation_phase == IslandRotationPhase.NONE, 0.0),
        (disruption_phase == DisruptionPhase.NONE, default_Wdot),
        (disruption_phase == DisruptionPhase.TQ, TQ_Wdot),
        (disruption_phase == DisruptionPhase.CQ, CQ_Wdot),
    ]

    Wdot = select_w_tuples(conditions_to_choices, default=default_Wdot)
    return Wdot

def calculate_rotation_dot(rotation_phase: IslandRotationPhase, q2_rot_freq: float, rot_dur: float, locking_dur: float, mode_number: IslandModeNumber) -> ArrayLike:
    """Calculate the time derivative of the rotation frequency based on the rotation phase.

    Args:
        rotation_phase (IslandRotationPhase): The phase of the island rotation.
        q2_rot_freq (float): TODO(sweeney)
        rot_dur (float): TODO(sweeney)
        locking_dur (float): TODO(sweeney)

    Returns:
        ArrayLike: The time derivative of the rotation frequency.
    """

    initial_rot_freq = initial_rot_freq_dict[mode_number]

    conditions_and_choices = [
        (rotation_phase == IslandRotationPhase.NONE, 0.0),
        (rotation_phase == IslandRotationPhase.ROTATING, -(initial_rot_freq / 2.0) / (rot_dur)),
        (rotation_phase == IslandRotationPhase.DECELERATING, -(initial_rot_freq / 2.0) / (locking_dur)),
        (rotation_phase == IslandRotationPhase.LOCKED, 0.0),
    ]

    Fdot = select_w_tuples(conditions_and_choices, default=0.0)
    return Fdot


@chex.dataclass
class Tearing(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.

        magx_time: Array  # deg, a time array on which to output the data
        thincurr_file: str  # path and filename of txt file defining the ThinCurr transfer functions
        ods_file: str  # path to the ODS object

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        W: dict[IslandModeNumber, float]  # m, Nxm
        F: dict[IslandModeNumber, float]  # Hz
        wave_phase: dict[IslandModeNumber, float]  # rad

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        q2_rot_freq: float
        rot_dur: float
        locking_dur: float
        disruption_phase: DisruptionPhase
        island_rotation_phase: IslandRotationPhase

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        params: "Params"  # noqa: F821

    config: Config
    phi_probes: Array  # deg, toroidal locations of probes
    phi_sens: Array  # deg, toroidal locations of sensors

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        disruption_phase = round(params.disruption_phase)
        island_rotation_phase = round(params.island_rotation_phase)

        Wdot = {mode: calculate_tearing_growth_rate(disruption_phase, island_rotation_phase, mode) for mode in IslandModeNumber}
        Fdot = {mode: calculate_rotation_dot(island_rotation_phase, params.q2_rot_freq, params.rot_dur, params.locking_dur, mode) for mode in IslandModeNumber}
        wave_phase_dot = {mode: state.F[mode]*2*jnp.pi for mode in IslandModeNumber}

        for mode in IslandModeNumber:
            # Force W to stay positive
            width_operand = state.W[mode]
            state.W[mode] = lax.cond(width_operand > 0, lambda x: x, lambda x: 0.0, width_operand)
            # If mode is born, set F to initial value
            state_operand = island_rotation_phase
            state.F[mode] = lax.cond(state_operand == IslandRotationPhase.SPAWN, lambda x: initial_rot_freq_dict[mode], lambda x: state.F[mode], state_operand)

        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot, wave_phase=wave_phase_dot)
        # Make an output.
        out = Tearing.Output(state_dot=state_dot, params=params)
        return state_dot, out
    