from enum import IntEnum

import chex
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

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
    ROTATING = 1
    DECELERATING = 2
    LOCKED = 3


def calculate_tearing_growth_rate(disruption_phase: DisruptionPhase) -> ArrayLike:
    """Calculate the tearing growth rate based on the disruption phase.
    This currently returns hard-coded values for the growth rate, but in the future
    this could be replaced with a more sophisticated model.

    Args:
        disruption_phase (DisruptionPhase): The phase of the disruption.

    Returns:
        ArrayLike: The growth rate of the tearing mode.
    """
    default_Wdot = jnp.array(4e-2 / 0.5)  # m/s
    TQ_Wdot = jnp.array(4e-2 / 0.002)  # m/s
    CQ_Wdot = jnp.array(-4e-2 / 0.005)  # m/s

    # This is a somewhat annoying Jax construct: if-statements aren't always kosher.
    # In this case, we can use the numpy-like jnp.select to choose the appropriate array.
    conditions_to_choices = [
        (disruption_phase == DisruptionPhase.NONE, default_Wdot),
        (disruption_phase == DisruptionPhase.TQ, TQ_Wdot),
        (disruption_phase == DisruptionPhase.CQ, CQ_Wdot),
    ]

    Wdot = select_w_tuples(conditions_to_choices, default=default_Wdot)
    return Wdot


def calculate_rotation_dot(rotation_phase: IslandRotationPhase, q2_rot_freq: float, rot_dur: float, locking_dur: float) -> ArrayLike:
    """Calculate the time derivative of the rotation frequency based on the rotation phase.

    Args:
        rotation_phase (IslandRotationPhase): The phase of the island rotation.
        q2_rot_freq (float): TODO(sweeney)
        rot_dur (float): TODO(sweeney)
        locking_dur (float): TODO(sweeney)

    Returns:
        ArrayLike: The time derivative of the rotation frequency.
    """
    conditions_and_choices = [
        (rotation_phase == IslandRotationPhase.NONE, 0.0),
        (rotation_phase == IslandRotationPhase.ROTATING, -(q2_rot_freq / 2.0) / (rot_dur)),
        (rotation_phase == IslandRotationPhase.DECELERATING, -(q2_rot_freq / 2.0) / (locking_dur)),
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
        W: float  # m, Nxm
        F: float  # Hz

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

        Wdot = calculate_tearing_growth_rate(disruption_phase)
        Fdot = calculate_rotation_dot(island_rotation_phase, params.q2_rot_freq, params.rot_dur, params.locking_dur)
        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot)
        # Make an output.
        out = Tearing.Output(state_dot=state_dot, params=params)
        return state_dot, out
