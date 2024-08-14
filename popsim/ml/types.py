import typing

import chex
import equinox as eqx
import optax
from jaxtyping import Array

from popsim import ModuleBase
from popsim.simulate import SimInput

# Define modules that are considered "trainable".
TrainableModule = typing.Union[eqx.Module, ModuleBase]
ModulePartitionFunc = typing.Callable[[TrainableModule], tuple[TrainableModule, TrainableModule]]


@chex.dataclass
class TrainState:
    step: int
    epoch: int
    opt_state: optax.OptState
    model: TrainableModule

    @classmethod
    def create_new(
        cls,
        model: TrainableModule,
        optimizer: optax.GradientTransformation,
        partition_fn: typing.Callable[[TrainableModule], tuple[TrainableModule, TrainableModule]],
    ):
        trainable, _ = partition_fn(model)
        opt_state = optimizer.init(trainable)
        return cls(0, 0, opt_state, model)


@chex.dataclass
class EvalInput:
    """Input data structure for evaluating a module."""

    sim_input: SimInput  # Standard input data structure for running a simulation.
    targets: dict[str, Array]  # Dict of target variables to compare module outputs against.
