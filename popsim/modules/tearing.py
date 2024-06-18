import typing
from enum import IntEnum

import chex
import jax.numpy as jnp
from jaxtyping import Array

from popsim import ModuleBase

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


def select_dict(conditions_to_choices: dict, default: typing.Any = 0.0):
    conditions = jnp.array(list(conditions_to_choices.keys()))
    choices = jnp.array(list(conditions_to_choices.values()))
    return jnp.select(conditions, choices, default=default)


def calculate_tearing_growth_rate(disruption_phase: DisruptionPhase) -> Array:
    """Calculate the tearing growth rate based on the disruption phase.
    This currently returns hard-coded values for the growth rate, but in the future
    this could be replaced with a more sophisticated model.

    Args:
        disruption_phase (DisruptionPhase): The phase of the disruption.

    Returns:
        Array: The growth rate of the tearing mode.
    """
    default_Wdot = jnp.array(4e-2 / 0.5)  # m/s
    TQ_Wdot = jnp.array(4e-2 / 0.002)  # m/s
    CQ_Wdot = jnp.array(-4e-2 / 0.005)  # m/s

    # This is a somewhat annoying Jax construct: if-statements aren't always kosher.
    # In this case, we can use the numpy-like jnp.select to choose the appropriate array.
    conditions_to_choices = {
        DisruptionPhase.NONE == disruption_phase: default_Wdot,
        DisruptionPhase.TQ == disruption_phase: TQ_Wdot,
        DisruptionPhase.CQ == disruption_phase: CQ_Wdot,
    }

    # Use jnp.select to choose the appropriate value.
    return select_dict(conditions_to_choices, default=default_Wdot)


def calculate_rotation_dot(rotation_phase: IslandRotationPhase, q2_rot_freq: float, rot_dur: float, locking_dur: float) -> Array:
    """Calculate the time derivative of the rotation frequency based on the rotation phase.

    Args:
        rotation_phase (IslandRotationPhase): The phase of the island rotation.
        q2_rot_freq (float): TODO(sweeney)
        rot_dur (float): TODO(sweeney)
        locking_dur (float): TODO(sweeney)

    Returns:
        Array: The time derivative of the rotation frequency.
    """

    conditions_to_choices = {
        IslandRotationPhase.NONE == rotation_phase: 0.0,
        IslandRotationPhase.ROTATING == rotation_phase: -(q2_rot_freq / 2.0) / (rot_dur),  # Hz/s
        IslandRotationPhase.DECELERATING == rotation_phase: -(q2_rot_freq / 2.0) / (locking_dur),  # Hz/s
        IslandRotationPhase.LOCKED == rotation_phase: 0.0,
    }

    return select_dict(conditions_to_choices, default=0.0)


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
