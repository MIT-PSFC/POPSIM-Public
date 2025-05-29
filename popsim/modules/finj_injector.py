import chex

from popsim import ModuleBase

# TODO: Include delays (should be cascade of FOPDT)


class FinjInjector(ModuleBase):
    """A model of a FINJ injector."""

    @chex.dataclass
    class Config:
        gas_temperature: float = 293  # [K]
        k_B: float = 1.380649e-23  # [J/K]

    @chex.dataclass
    class State:
        valve_flow_rate: float  # [Pa m^3/s]
        pipe_flow_rate: float  # [Pa m^3/s]

    @chex.dataclass
    class Output:
        valve_flow_rate: float  # [Pa m^3/s]
        valve_flow_rate_n_per_s: float  # [#/s]

    @chex.dataclass
    class Inputs:
        flow_rate_command: float  # [Pa m^3/s]
        valve_flow_rate_tau: float  # [s]
        pipe_flow_rate_tau: float  # [s]

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        """Derivatives of the valve and pipe flow rates in response to flow rate commands"""
        valve_flow_rate_dot = (inputs.flow_rate_command - state.valve_flow_rate) / inputs.valve_flow_rate_tau
        pipe_flow_rate_dot = (state.valve_flow_rate - state.pipe_flow_rate) / inputs.pipe_flow_rate_tau
        # Convert flow rate to #/s
        valve_flow_rate_n_per_s = state.valve_flow_rate * self.config.k_B * self.config.gas_temperature

        return FinjInjector.State(valve_flow_rate=valve_flow_rate_dot, pipe_flow_rate=pipe_flow_rate_dot), FinjInjector.Output(
            valve_flow_rate=state.valve_flow_rate, valve_flow_rate_n_per_s=valve_flow_rate_n_per_s
        )
