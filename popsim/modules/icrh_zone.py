import chex
import jax

from popsim import ModuleBase

"""
A model of an ICRH zone.
"""


@chex.dataclass
class IcrhZone(ModuleBase):
    @chex.dataclass
    class Config:
        reflected_power_ratio: float = 0.1  # [-]

    @chex.dataclass
    class State:
        """
        static: float = 1.0  # [-]

        @property
        def is_static(self):
            return jnp.bool_(self.static == 1.0)
        """

    @chex.dataclass
    class Output:
        transmitted_power: float  # [W]
        reflected_power: float  # [W]

    # TODO: Look up tables of reflected power ratio vs frequency for different plasma conditions. Should this be config?

    @chex.dataclass
    class Params:
        frequency_command: float  # RF frequency [MHz]
        power_command: float  # Zone power command [W]

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params, key: jax.random.PRNGKey = None) -> tuple[State, Output]:
        state_dot = IcrhZone.State()  # IcrhZone.State(static=0.0)  # state_dot = 0.0
        out = IcrhZone.Output(
            transmitted_power=(1.0 - self.config.reflected_power_ratio) * params.power_command,
            reflected_power=self.config.reflected_power_ratio * params.power_command,
        )
        return state_dot, out
