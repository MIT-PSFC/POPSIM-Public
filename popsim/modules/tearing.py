from enum import IntEnum
from types import MappingProxyType
from typing import Optional

import chex
import jax.numpy as jnp
import numpy as np
from jax import lax
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


class Island:
    # Hard-coded values for the island growth rates
    _DEFAULT_WDOT_DICT = MappingProxyType(
        {
            (3, 2): 4e-2 / 0.5,  # m/s
            (2, 1): 10e-2 / 0.5,
            (3, 1): 3e-2 / 0.5,
        }
    )
    _TQ_WDOT_DICT = MappingProxyType(
        {
            (3, 2): 4e-2 / 0.002,  # m/s
            (2, 1): 10e-2 / 0.002,
            (3, 1): 3e-2 / 0.002,
        }
    )
    _CQ_WDOT_DICT = MappingProxyType(
        {
            (3, 2): -4e-2 / 0.005,  # m/s
            (2, 1): -10e-2 / 0.005,
            (3, 1): -3e-2 / 0.005,
        }
    )

    # Hard-coded values for the initial rotation frequency of the island
    _INITIAL_ROT_FREQ_DICT = MappingProxyType(
        {
            (3, 2): 7e3 * 1.5,
            (2, 1): 7e3,
            (3, 1): 7e3 * 0.67,
        }
    )

    def __init__(
        self,
        m: int,
        n: int,
        default_Wdot: Optional[float] = None,
        TQ_Wdot: Optional[float] = None,
        CQ_Wdot: Optional[float] = None,
        initial_rot_freq: Optional[float] = None,
    ):
        """Initialize an Island object with a given poloidal and toroidal mode number.

        Args:
            m (int): The poloidal mode number.
            n (int): The toroidal mode number.
            default_Wdot (float, optional): The default growth rate of the island. Defaults to hardcoded value.
            TQ_Wdot (float, optional): The growth rate of the island during a TQ disruption. Defaults to hardcoded value.
            CQ_Wdot (float, optional): The growth rate of the island during a CQ disruption. Defaults to hardcoded value.
            initial_rot_freq (float, optional): The initial rotation frequency of the island. Defaults to hardcoded value.
        """

        self.m = m
        self.n = n

        self.default_Wdot = default_Wdot
        self.TQ_Wdot = TQ_Wdot
        self.CQ_Wdot = CQ_Wdot
        self.initial_rot_freq = initial_rot_freq

        if initial_rot_freq is None:
            self.initial_rot_freq = self._INITIAL_ROT_FREQ_DICT[(m, n)]
        if default_Wdot is None:
            self.default_Wdot = self._DEFAULT_WDOT_DICT[(m, n)]
        if TQ_Wdot is None:
            self.TQ_Wdot = self._TQ_WDOT_DICT[(m, n)]
        if CQ_Wdot is None:
            self.CQ_Wdot = self._CQ_WDOT_DICT[(m, n)]

    def __str__(self) -> str:
        return f"Island({self.m},{self.n})"

    def __eq__(self, other) -> bool:
        return self.m == other.m and self.n == other.n

    def __gt__(self, other) -> bool:
        return (self.m / self.n) > (other.m / other.n)

    def __lt__(self, other) -> bool:
        return (self.m / self.n) < (other.m / other.n)

    def __hash__(self) -> int:
        return hash((self.m, self.n))


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


def calculate_tearing_growth_rate(disruption_phase: DisruptionPhase, rotation_phase: TearingPhase, island: Island) -> ArrayLike:
    """Calculate the tearing growth rate based on its rotation phase and the disruption phase.
    This presently returns hard-coded values for the growth rate, but in the future
    this could be replaced with a more sophisticated model.

    Args:
        disruption_phase (DisruptionPhase): The phase of the disruption.
        rotation_phase (TearingPhase): The phase of the island rotation.
        island (Island): The island for which to calculate the growth rate.

    Returns:
        ArrayLike: The growth rate of the tearing mode.
    """

    conditions_to_choices = [
        (rotation_phase == TearingPhase.NONE, 0.0),
        (disruption_phase == DisruptionPhase.NONE, island.default_Wdot),
        (disruption_phase == DisruptionPhase.TQ, island.TQ_Wdot),
        (disruption_phase == DisruptionPhase.CQ, island.CQ_Wdot),
    ]

    Wdot = select_w_tuples(conditions_to_choices, default=island.default_Wdot)
    return Wdot


def calculate_rotation_dot(rotation_phase: TearingPhase, rot_dur: float, locking_dur: float, island: Island) -> ArrayLike:
    """Calculate the time derivative of the rotation frequency based on the rotation phase.

    Args:
        rotation_phase (TearingPhase): The phase of the island rotation.
        rot_dur (float): How long the island rotates before beginning deceleration.
        locking_dur (float): How long the island decelerates before locking.
        island (Island): The island for which to calculate the rotation frequency time derivative.

    Returns:
        ArrayLike: The time derivative of the rotation frequency.
    """

    conditions_and_choices = [
        (rotation_phase == TearingPhase.NONE, 0.0),
        (rotation_phase == TearingPhase.ROTATING, -(island.initial_rot_freq / 2.0) / (rot_dur)),
        (rotation_phase == TearingPhase.DECELERATING, -(island.initial_rot_freq / 2.0) / (locking_dur)),
        (rotation_phase == TearingPhase.LOCKED, 0.0),
    ]

    Fdot = select_w_tuples(conditions_and_choices, default=0.0)
    return Fdot


@chex.dataclass
class Tearing(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the data that configures the module and will be static during the simulation.
        magx_time: Array  # deg, a time array on which to output the data
        islands: list[Island]

    @chex.dataclass
    class State:
        # Define the differential state variables that will be integrated during the simulation.
        # If a variable is defined in here, then the module must output its time derivative in the __call__ method.
        W: dict[Island, float]  # m, Nxm
        F: dict[Island, float]  # Hz
        mode_phase: dict[Island, float]  # rad

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        rot_dur: float
        locking_dur: float
        disruption_phase: DisruptionPhase
        tearing_phase: TearingPhase
        cur_per_W: float = 1e3 / 1e-2  # Perturbed current per island width [A/m] TODO(ZanderKeith) a guess for now

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        params: "Params"  # noqa: F821
        mode_current: dict[Island, float]  # A
        mode_phase: dict[Island, float]  # rad
        mode_freq: dict[Island, float]  # Hz

    config: Config

    def __init__(self, config):
        self.config = config
        self.islands = islands

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        disruption_phase = round(params.disruption_phase)
        tearing_phase = round(params.tearing_phase)

        Wdot = {island: calculate_tearing_growth_rate(disruption_phase, tearing_phase, island) for island in self.islands}
        Fdot = {island: calculate_rotation_dot(tearing_phase, params.rot_dur, params.locking_dur, island) for island in self.islands}
        mode_phase_dot = {island: state.F[island] * 2 * jnp.pi for island in self.islands}

        for mode in self.islands:
            # Force W to stay positive
            width_operand = state.W[mode]
            state.W[mode] = lax.cond(width_operand > 0, lambda x: x, lambda x: 0.0, width_operand)

            # If mode is born, set F to initial value
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.SPAWN, mode.initial_rot_freq, state.F[mode])
            # If mode is locked, set F to 0
            state.F[mode] = jnp.where(tearing_phase == TearingPhase.LOCKED, 0.0, state.F[mode])

        # Calculate perturbed current
        perturbed_current = {island: state.W[island] * params.cur_per_W for island in self.islands}

        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot)
        # Make an output.
        out = Tearing.Output(
            state_dot=state_dot, params=params, mode_current=perturbed_current, mode_phase=state.mode_phase, mode_freq=state.F
        )
        return state_dot, out
