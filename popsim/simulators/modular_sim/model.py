import chex
import equinox as eqx

# NOTE: We may need an add_enum command for the generator to know which of these are needed
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree

from popsim import ModuleBase
from popsim.enums import (
    Species,
    SpeciesContainer,
)
from popsim.modules.density import Density
from popsim.modules.finj_injector import FinjInjector
from popsim.modules.icrh_zone import IcrhZone
from popsim.modules.power_balance import PowerBalance


@chex.dataclass
class ModularModel(ModuleBase):
    @chex.dataclass
    class State:
        """
        State variables for the ModularSim model.
        """

        power_balance_state: PowerBalance.State
        density_state: Density.State
        icrh_zone_state: IcrhZone.State
        finj_state: dict[Species, FinjInjector.State]

    @chex.dataclass
    class Params:
        """
        Dynamic parameters for the ModularSim model.
        """

        confinement_time_scalar: float  # [-]
        confinement_time: float  # [-]
        P_aux_MW: float  # [MW]
        fueling19: dict[Species, float]  # 1e19/s
        particle_confinement_scalar: dict[Species, float]  # [-]
        valve_flow_rate_tau: float  # [s]
        pipe_flow_rate_tau: float  # [s]
        volume_dot: float  # [m^3/s]
        volume: float  # [m^3]

    @chex.dataclass
    class Config:
        """
        Static compile-time configuration for the ModularSim model.
        """

        species: SpeciesContainer
        icrh_zone: IcrhZone.Config
        finj_injectors: dict[Species, FinjInjector.Config]
        power_balance: PowerBalance.Config
        density: Density.Config

    @chex.dataclass
    class Output:
        locals: PyTree[ArrayLike]

    config: Config
    icrh_zone_module: IcrhZone
    finj_injector_modules: dict[Species, FinjInjector]
    power_balance_module: PowerBalance
    density_module: Density

    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.icrh_zone_module = IcrhZone(config=self.config.icrh_zone)
        self.finj_injector_modules = {k: FinjInjector(config=self.config.finj_injectors[k]) for k in self.config.finj_injectors.keys()}
        self.power_balance_module = PowerBalance(config=self.config.power_balance)
        self.density_module = Density(config=self.config.density)

    def __call__(self, state: State, params: Params, return_aux: bool = False) -> tuple[State, Output]:
        icrh_zone_params = IcrhZone.Params(
            frequency_command=120,
            power_command=params.P_aux_MW,
        )
        icrh_zone_dot, icrh_zone_output = self.icrh_zone_module(state.icrh_zone_state, icrh_zone_params)

        finj_injector_output = {}
        finj_injector_dot = {}
        for k, v in params.fueling19.items():
            finj_injector_params = FinjInjector.Params(
                flow_rate_command=v,  # [Pa m^3/s]
                valve_flow_rate_tau=params.valve_flow_rate_tau,  # [s]
                pipe_flow_rate_tau=params.pipe_flow_rate_tau,  # [s]
            )

            finj_injector_dot[k], finj_injector_output[k] = self.finj_injector_modules[k](state.finj_state[k], finj_injector_params)

        power_balance_params = PowerBalance.Params(
            P_aux=icrh_zone_output.transmitted_power,  # Auxilliary power [MW]
            confinement_time=params.confinement_time,  # energy confinement time in seconds
        )

        power_balance_dot, power_balance_output = self.power_balance_module(state.power_balance_state, power_balance_params)

        sources_and_sinks = {k: {} for k in self.config.species.species}
        for k in params.fueling19.keys():
            sources_and_sinks[k]["fueling19"] = finj_injector_output[k]["valve_flow_rate_n_per_s"] * 1e-19

        density_params = Density.Params(
            sources_and_sinks=sources_and_sinks,
            species_confinement_time=jax.tree.map(lambda k: k * params.confinement_time, params.particle_confinement_scalar),
            volume_dot=params.volume_dot,  # TODO(allenw): add with time-varying geometry.
            volume=params.volume,  # params.geometry.plasma_volume,
        )

        density_dot, density_out = self.density_module(state.density_state, density_params)

        state_dot = ModularModel.State(  # NOTE: The generator would build this
            power_balance_state=power_balance_dot,
            density_state=density_dot,
            icrh_zone_state=icrh_zone_dot,
            finj_state=finj_injector_dot,
        )

        # Filter out any non-array-like variables
        aux_data = eqx.filter(locals(), eqx.is_array_like)
        # Promote any scalar-like variables to arrays
        aux_data = jax.tree.map(jnp.asarray, aux_data)
        out = ModularModel.Output(locals=aux_data)
        return state_dot, out
