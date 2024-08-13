import jax
import jax.numpy as jnp
import numpy as np


def jax_to_numpy_array(arr: jnp.ndarray) -> np.ndarray:
    """Convert a JAX array to a numpy array. Note that most of the time, the conversion will happen automatically.
    This function primarily exists as a one stop shop for handling edge cases. For example, PRNGKeyArray uses a custom
    dtype, which is not handled by the automatic conversion.

    Args:
        arr (jnp.ndarray): The JAX array to convert.

    Returns:
        np.ndarray: The numpy array.
    """

    if jax.dtypes.issubdtype(arr.dtype, jax.dtypes.prng_key):
        arr = jax.random.key_data(arr)

    return arr
