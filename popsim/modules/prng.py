import random

import chex
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from popsim import ModuleBase, discrete_time_field


def random_seed():
    return random.randint(jnp.iinfo(jnp.int32).min, jnp.iinfo(jnp.int32).max)


class PRNGModule(ModuleBase):
    """A module for generating a time-dependent trajectory of PRNGKeyArray.

    Note that the state is the "seed", not the key itself. This is because the PRNGKeyArray can be an array of special types, which is not conducive for re-initialization.
    """

    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        seed: int = discrete_time_field(
            default_factory=random_seed
        )  # Integer seed for generating a PRNGKeyArray that gets updated at every time step. If the user does not provide a seed, a random seed is generated.

    @chex.dataclass
    class Params:
        pass

    @chex.dataclass
    class Output:
        key: PRNGKeyArray  # Generated PRNGKeyArray that can be used, for example, in jax.random functions.

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        # Generate a new key from the current seed.
        key = jax.random.key(state.seed)

        # Generate the seed for the next iteration.
        next_seed = jax.random.randint(key, (1,), minval=jnp.iinfo(jnp.int32).min, maxval=jnp.iinfo(jnp.int32).max).squeeze()

        return self.State(seed=next_seed), self.Output(key=key)
