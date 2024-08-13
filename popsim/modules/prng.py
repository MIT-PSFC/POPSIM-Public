import chex
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from popsim import ModuleBase, discrete_time_field


class PRNGModule(ModuleBase):
    """A module for generating a time-dependent trajectory of PRNGKeyArray.

    Note that the state is the "seed", not the key itself. This is because the PRNGKeyArray can be an array of special types, which is not conducive for re-initialization.
    """

    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        seed: int = discrete_time_field()  # Integer seed for generating a PRNGKeyArray.

    @chex.dataclass
    class Params:
        pass

    @chex.dataclass
    class Output:
        key_array: PRNGKeyArray  # Generated PRNGKeyArray that can be used, for example, in jax.random functions.

    def __call__(self, state: State, params: Output) -> tuple[State, Output]:
        # Generate a new key from the current seed.
        key = jax.random.key(state.seed)

        # Generate the seed for the next iteration.
        next_seed = jax.random.randint(key, (1,), minval=jnp.iinfo(jnp.int32).min, maxval=jnp.iinfo(jnp.int32).max).squeeze()

        return self.State(seed=next_seed), self.Output(key_array=key)
