from abc import abstractmethod

import chex
import diffrax
import equinox as eqx
import xarray as xr
from jaxtyping import Array, ArrayLike, PyTree

from popsim import ModuleBase, interp
from popsim.ml.utils import _repeat_time_hack
from popsim.sim_utils import SimInput
from popsim.simulate import _diffrax_simulate

"""
Base classes for defining environments for training and evaluating modules.
"""


@chex.dataclass
class ModuleEvalEnvInput:
    initial_state: dict[str, ArrayLike] | xr.Dataset
    inputs: dict[str, ArrayLike] | xr.Dataset
    time: Array


@eqx.filter_jit
def call_module_eval_env(env: "ModuleEvalEnv", env_input: ModuleEvalEnvInput) -> diffrax.Solution:
    """Given a ModuleEvalEnvInput object consisting of arrays:
        1) Create State and Inputs objects from the arrays.
        2) Interpolate the Inputs.
        3) Simulate the module and return a diffrax.Solution object.

    Args:
        env_input (ModuleEvalEnvInput):

    Returns:
        diffrax.Solution: the solution.
    """

    # Construct the initial State and Input objects.
    if isinstance(env_input.initial_state, dict) and isinstance(env_input.inputs, dict):
        create_state_input = env_input.initial_state | env_input.inputs
    elif isinstance(env_input.initial_state, xr.Dataset) and isinstance(env_input.inputs, xr.Dataset):
        create_state_input = xr.merge([env_input.initial_state, env_input.inputs])
    else:
        raise ValueError("Invalid input types for initial_state and inputs")
    initial_state = env.create_state(create_state_input)

    inputs = env.create_inputs(env_input.inputs)

    time = _repeat_time_hack(env_input.time)

    inputs_interped = interp.interp(time, inputs, interp.InterpType.RECTILINEAR)

    sim_input = SimInput(
        time=time,
        initial_state=initial_state,
        inputs=inputs_interped,
    )

    sol = _diffrax_simulate(env.module, sim_input)
    return sol


class ModuleEvalEnv(eqx.Module):
    module: ModuleBase

    @staticmethod
    @abstractmethod
    def create_state(data: dict[str, ArrayLike]) -> "State":  # noqa: F821
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_inputs(data: dict[str, ArrayLike]) -> "Inputs":  # noqa: F821
        raise NotImplementedError

    @eqx.filter_jit
    def __call__(self, env_input: ModuleEvalEnvInput) -> diffrax.Solution:
        return call_module_eval_env(self, env_input)


class ModuleTrainingEnv(ModuleEvalEnv):
    @abstractmethod
    def get_trainable(self) -> PyTree | tuple[PyTree]:
        raise NotImplementedError
