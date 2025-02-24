from abc import abstractmethod

import chex
import diffrax
import equinox as eqx
import xarray as xr
from jaxtyping import Array, ArrayLike, PyTree

from popsim import ModuleBase, interp
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
def call_module_eval_env(env: "ModuleEvalEnv", env_input: ModuleEvalEnvInput, max_step_mult: int = 2) -> diffrax.Solution:
    """Run a ModuleEvalEnv with the given input. This handles state initialization and interpolation of time-dependent inputs.

    Args:
        env (ModuleEvalEnv): the environment to run.
        env_input (ModuleEvalEnvInput): the input to the environment.
        max_step_mult (int, optional): the maximum number of diffeqsolve steps is the number of time steps in env_input times this quantity. Defaults to 2.

    Raises:
        ValueError: if the input types for initial_state and inputs are invalid.

    Returns:
        diffrax.Solution: the solution of the simulation.
    """
    # Construct the initial State and Input objects.
    if not (
        (isinstance(env_input.initial_state, dict) and isinstance(env_input.inputs, dict))
        or (isinstance(env_input.initial_state, xr.Dataset) and isinstance(env_input.inputs, xr.Dataset))
    ):
        raise ValueError("Invalid input types for initial_state and inputs")
    initial_state = env.create_state(env_input.initial_state, env_input.inputs)

    inputs = env.create_inputs(env_input.inputs)

    time = env_input.time

    inputs_interped = interp.interp(time, inputs, interp.InterpType.RECTILINEAR)

    sim_input = SimInput(
        time=time,
        initial_state=initial_state,
        inputs=inputs_interped,
    )

    max_steps = max_step_mult * time.size
    sol = _diffrax_simulate(env.module, sim_input, max_steps=max_steps)
    return sol


class ModuleEvalEnv(eqx.Module):
    module: ModuleBase

    @staticmethod
    @abstractmethod
    def create_state(observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]) -> "State":  # noqa: F821
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def create_inputs(inputs: dict[str, ArrayLike]) -> "Inputs":  # noqa: F821
        raise NotImplementedError

    @eqx.filter_jit
    def __call__(self, env_input: ModuleEvalEnvInput) -> diffrax.Solution:
        return call_module_eval_env(self, env_input)


class ModuleTrainingEnv(ModuleEvalEnv):
    @abstractmethod
    def get_trainable(self) -> PyTree | tuple[PyTree]:
        raise NotImplementedError
