from enum import IntEnum
from typing import Optional

import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from popsim import ModuleBase
from popsim.logic_utils import select_w_tuples

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
    (3, 1): 7e3 * 0.67,
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
        magx_time: Array  # deg, a time array on which to output the data
        modes: list[tuple]

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        W: dict[tuple, float]  # m, Nxm
        F: dict[tuple, float]  # Hz
        mode_phase: dict[tuple, float]  # rad

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        rot_dur: float
        locking_dur: float
        disruption_phase: DisruptionPhase
        tearing_phase: TearingPhase
        cur_per_W: float = 1e3 / 1e-2  # Perturbed current per island width [A/m] TODO(ZanderKeith) a guess for now
        default_wdot: dict[tuple, float] = DEFAULT_WDOT
        tq_wdot: dict[tuple, float] = TQ_WDOT
        cq_wdot: dict[tuple, float] = CQ_WDOT
        initial_rot_freq: dict[tuple, float] = INITIAL_ROT_FREQ

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        params: "Params"  # noqa: F821
        mode_current: dict[tuple, float]  # A
        mode_phase: dict[tuple, float]  # rad
        mode_freq: dict[tuple, float]  # Hz

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        disruption_phase = round(params.disruption_phase)
        tearing_phase = round(params.tearing_phase)

        Wdot = {
            mode: calculate_tearing_growth_rate(
                disruption_phase, tearing_phase, params.default_wdot[mode], params.tq_wdot[mode], params.cq_wdot[mode]
            )
            for mode in self.config.modes
        }
        Fdot = {
            mode: calculate_rotation_dot(tearing_phase, params.rot_dur, params.locking_dur, params.initial_rot_freq[mode])
            for mode in self.config.modes
        }
        mode_phase_dot = {mode: state.F[mode] * 2 * jnp.pi for mode in self.config.modes}

        for mode in self.config.modes:
            # Force W to stay positive
            state.W[mode] = jnp.where(state.W[mode] > 0, state.W[mode], 0.0)

            # If mode just spawned, set F to initial value
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.SPAWN, params.initial_rot_freq[mode], state.F[mode])
            # If mode is locked, set F to 0
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.LOCKED, 0.0, state.F[mode])

        # Calculate perturbed current
        perturbed_current = {mode: state.W[mode] * params.cur_per_W for mode in self.config.modes}

        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot)
        # Make an output.
        out = Tearing.Output(
            state_dot=state_dot, params=params, mode_current=perturbed_current, mode_phase=state.mode_phase, mode_freq=state.F
        )
        return state_dot, out
