import chex

from popsim import TimeDepModule

"""
A power balance dynamics model that evolves stored energy.
"""


class PowerBalance(TimeDepModule):
    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        stored_energy: float  # [MJ]

    @chex.dataclass
    class Output:
        stored_energy: float  # [MJ]

    @chex.dataclass
    class Inputs:
        P_aux: float  # Power from auxillary heating [MW]
        confinement_time: float  # energy confinement time in seconds

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        stored_energy_dot = -state.stored_energy / inputs.confinement_time + inputs.P_aux
        return PowerBalance.State(stored_energy=stored_energy_dot), PowerBalance.Output(
            stored_energy=state.stored_energy,
        )
