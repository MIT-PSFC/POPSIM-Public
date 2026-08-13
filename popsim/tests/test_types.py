from jax import random
from jaxtyping import Array, PRNGKeyArray

from popsim.types import StaticSamplerFn


def test_static_sampler_fn():
    def sample_fn(key: PRNGKeyArray) -> Array:
        return random.uniform(key, (3,))
    
    assert isinstance(sample_fn, StaticSamplerFn)

    def sample_fn2(key) -> Array:
        return random.uniform(key, (3,))
    
    assert isinstance(sample_fn2, StaticSamplerFn)
