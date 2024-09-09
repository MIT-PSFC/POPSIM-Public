import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

"""
Collection of patches to work around issues in packages we depend on.
"""


def patched_error_if(x, pred, msg):
    # The error_if function is not happy if we input prng_key types.
    # This is due to its calling of jax.pure_callback, which tries to use a pure numpy check on the array which has a custom JAX type.
    x = eqx.filter(x, lambda leaf: isinstance(leaf, Array) and not jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key))

    # Provide a sentinel x.
    # Work-around until https://github.com/patrick-kidger/equinox/issues/835 is resolved.
    flat = jax.tree.leaves(x)
    if len(flat) == 0:
        sentinel_x = jnp.array(True)
        return eqx.error_if(sentinel_x, pred, msg)
    else:
        return eqx.error_if(x, pred, msg)
