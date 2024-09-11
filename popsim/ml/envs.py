from abc import ABC, abstractmethod

import chex
import diffrax
import equinox as eqx
from jaxtyping import Array, ArrayLike

from popsim import ModuleBase, interp
from popsim.ml.utils import _repeat_time_hack
from popsim.sim_utils import SimInput
from popsim.simulate import _diffrax_simulate

"""
Base classes for defining environments for training and evaluating modules.
"""


@chex.dataclass
class ModuleEvalEnvInput:
    initial_state: dict[str, ArrayLike]
    params: dict[str, ArrayLike]
    time: Array


@eqx.filter_jit
def call_module_eval_env(env: "ModuleEvalEnv", env_input: ModuleEvalEnvInput) -> diffrax.Solution:
    """Given a ModuleEvalEnvInput object consisting of arrays:
        1) Create State and Params objects from the arrays.
        2) Interpolate the Params.
        3) Simulate the module and return a diffrax.Solution object.

    Args:
        env_input (ModuleEvalEnvInput):

    Returns:
        diffrax.Solution: the solution.
    """
    # Construct the initial State and Params objects.
    create_state_input = env_input.initial_state | env_input.params
    initial_state = env.create_state(create_state_input)
    params = env.create_params(env_input.params)

    params_interped = interp.interp(_repeat_time_hack(env_input.time), params, interp.InterpType.RECTILINEAR)

    sim_input = SimInput(
        time=env_input.time,
        initial_state=initial_state,
        params=params_interped,
    )

    sol = _diffrax_simulate(env.module, sim_input)
    return sol


class ModuleEvalEnv(ABC):
    module: ModuleBase

    def __init__(self, module: ModuleBase):
        self.module = module

    @staticmethod
    @abstractmethod
    def create_state(data: dict[str, ArrayLike]) -> "State":  # noqa: F821
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_params(data: dict[str, ArrayLike]) -> "Params":  # noqa: F821
        raise NotImplementedError

    @eqx.filter_jit
    def __call__(self, env_input: ModuleEvalEnvInput) -> diffrax.Solution:
        return call_module_eval_env(self, env_input)
