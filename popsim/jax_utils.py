import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree


def leaves_as_array(tree: PyTree[ArrayLike]) -> Array:
    return jnp.atleast_1d(jnp.array(jax.tree_util.tree_leaves(tree)))
