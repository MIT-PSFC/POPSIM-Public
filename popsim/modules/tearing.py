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

class Island:

    def __init__(self, m, n, default_Wdot=None, TQ_Wdot=None, CQ_Wdot=None, initial_rot_freq=None):
        """ Initialize an Island object with the given mode number.

        Args:
            m (int): The poloidal mode number.
            n (int): The toroidal mode number.
            initial_rot_freq (float, optional): The initial rotation frequency of the island. Defaults to hardcoded value.
            default_Wdot (float, optional): The default growth rate of the island. Defaults to hardcoded value.
            TQ_Wdot (float, optional): The growth rate of the island during a TQ disruption. Defaults to hardcoded value.
            CQ_Wdot (float, optional): The growth rate of the island during a CQ disruption. Defaults to hardcoded value.
        """

        self.m = m
        self.n = n

        self.default_Wdot = default_Wdot
        self.TQ_Wdot = TQ_Wdot
        self.CQ_Wdot = CQ_Wdot
        self.initial_rot_freq = initial_rot_freq

        if initial_rot_freq is None:
            self.initial_rot_freq = self.initial_rot_freq_dict[(m, n)]
        if default_Wdot is None:
            self.default_Wdot = self.default_Wdot_dict[(m, n)]
        if TQ_Wdot is None:
            self.TQ_Wdot = self.TQ_Wdot_dict[(m, n)]
        if CQ_Wdot is None:
            self.CQ_Wdot = self.CQ_Wdot_dict[(m, n)]

    def __str__(self) -> str:
        return f"Island({self.m},{self.n})"

    def __eq__(self, other) -> bool:
        return (self.m, self.n) == (other.m, other.n)
    
    def __gt__(self, other) -> bool:
        return (self.m, self.n) > (other.m, other.n)
    
    def __lt__(self, other) -> bool:
        return (self.m, self.n) < (other.m, other.n)
    
    def __hash__(self) -> int:
        return hash((self.m, self.n))
        
    # Hard-coded values for the island growth rates
    default_Wdot_dict = {
        (3,2): 4e-2/0.5,  # m/s
        (2,1): 10e-2/0.5,
        (3,1): 3e-2/0.5
    }
    TQ_Wdot_dict = {
        (3,2): 4e-2/0.002,  # m/s
        (2,1): 10e-2/0.002,
        (3,1): 3e-2/0.002
    }
    CQ_Wdot_dict = {
        (3,2): -4e-2/0.005,  # m/s
        (2,1): -10e-2/0.005,
        (3,1): -3e-2/0.005
    }

    # Hard-coded values for the initial rotation frequency of the island
    initial_rot_freq_dict = {
        (3,2): 7e3*1.5,
        (2,1): 7e3,
        (3,1): 7e3*0.67,
    }

def calculate_tearing_growth_rate(disruption_phase: DisruptionPhase, rotation_phase: IslandRotationPhase, island: Island) -> ArrayLike:
    """Calculate the tearing growth rate based on its rotation phase and the disruption phase.
    This currently returns hard-coded values for the growth rate, but in the future
    this could be replaced with a more sophisticated model.

    Args:
        rotation_phase (IslandRotationPhase): The phase of the island rotation.

    Returns:
        ArrayLike: The growth rate of the tearing mode.
    """

    conditions_to_choices = [
        (rotation_phase == IslandRotationPhase.NONE, 0.0),
        (disruption_phase == DisruptionPhase.NONE, island.default_Wdot),
        (disruption_phase == DisruptionPhase.TQ, island.TQ_Wdot),
        (disruption_phase == DisruptionPhase.CQ, island.CQ_Wdot),
    ]

    Wdot = select_w_tuples(conditions_to_choices, default=island.default_Wdot)
    return Wdot

def calculate_rotation_dot(rotation_phase: IslandRotationPhase, rot_dur: float, locking_dur: float, island: Island) -> ArrayLike:
    """Calculate the time derivative of the rotation frequency based on the rotation phase.

    Args:
        rotation_phase (IslandRotationPhase): The phase of the island rotation.
        rot_dur (float): TODO(sweeney)
        locking_dur (float): TODO(sweeney)

    Returns:
        ArrayLike: The time derivative of the rotation frequency.
    """

    conditions_and_choices = [
        (rotation_phase == IslandRotationPhase.NONE, 0.0),
        (rotation_phase == IslandRotationPhase.ROTATING, -(island.initial_rot_freq / 2.0) / (rot_dur)),
        (rotation_phase == IslandRotationPhase.DECELERATING, -(island.initial_rot_freq / 2.0) / (locking_dur)),
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
        W: dict[Island, float]  # m, Nxm
        F: dict[Island, float]  # Hz
        mode_phase: dict[Island, float]  # rad

    @chex.dataclass
    class Params:
        # Define the, possibly time dependent, parameters that will be passed to the module.
        rot_dur: float
        locking_dur: float
        disruption_phase: DisruptionPhase
        island_rotation_phase: IslandRotationPhase

    @chex.dataclass
    class Output:
        state_dot: "State"  # noqa: F821
        params: "Params"  # noqa: F821
        mode_current: dict[Island, float]  # A
        mode_phase: dict[Island, float]  # rad
        mode_freq: dict[Island, float]  # Hz


    config: Config
    islands: list[Island]

    def __init__(self, config, islands: list[Island]):
        self.config = config
        self.islands = islands

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        disruption_phase = round(params.disruption_phase)
        island_rotation_phase = round(params.island_rotation_phase)

        Wdot = {island: calculate_tearing_growth_rate(disruption_phase, island_rotation_phase, island) for island in self.islands}
        Fdot = {island: calculate_rotation_dot(island_rotation_phase, params.rot_dur, params.locking_dur, island) for island in self.islands}
        mode_phase_dot = {island: state.F[island]*2*jnp.pi for island in self.islands}

        curPerW = 1e3/1e-2 # 1 kA/cm <- a guess for now

        for mode in self.islands:
            # Force W to stay positive
            width_operand = state.W[mode]
            state.W[mode] = lax.cond(width_operand > 0, lambda x: x, lambda x: 0.0, width_operand)

            # If mode is born, set F to initial value
            state_operand = island_rotation_phase
            state.F[mode] = lax.cond(state_operand == IslandRotationPhase.SPAWN, lambda x: mode.initial_rot_freq, lambda x: state.F[mode], state_operand)

            # Calculate perturbed current (just a guess for now)
            perturbed_current = {island: state.W[island]*curPerW for island in self.islands}

        # Make a state_dot.
        state_dot = Tearing.State(W=Wdot, F=Fdot, mode_phase=mode_phase_dot)
        # Make an output.
        out = Tearing.Output(
            state_dot=state_dot, 
            params=params, 
            mode_current=perturbed_current,
            mode_phase=state.mode_phase,
            mode_freq=state.F)
        return state_dot, out
    