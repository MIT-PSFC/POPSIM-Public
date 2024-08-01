import chex
import jax

from popsim import ModuleBase

"""
A model of a FINJ injector.
"""
# TODO: Include delays (should be cascade of FOPDT)


@chex.dataclass
class FinjInjector(ModuleBase):
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
    class Params:
        flow_rate_command: float  # [Pa m^3/s]
        valve_flow_rate_tau: float  # [s]
        pipe_flow_rate_tau: float  # [s]

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params, key: jax.random.PRNGKey = None) -> tuple[State, Output]:
        """Derivatives of the valve and pipe flow rates in response to flow rate commands"""
        valve_flow_rate_dot = (params.flow_rate_command - state.valve_flow_rate) / params.valve_flow_rate_tau
        pipe_flow_rate_dot = (state.valve_flow_rate - state.pipe_flow_rate) / params.pipe_flow_rate_tau
        # Convert flow rate to #/s
        valve_flow_rate_n_per_s = state.valve_flow_rate * self.config.k_B * self.config.gas_temperature

        return FinjInjector.State(valve_flow_rate=valve_flow_rate_dot, pipe_flow_rate=pipe_flow_rate_dot), FinjInjector.Output(
            valve_flow_rate=state.valve_flow_rate, valve_flow_rate_n_per_s=valve_flow_rate_n_per_s
        )
