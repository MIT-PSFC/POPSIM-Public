import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike


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

    return np.asarray(arr)


def min_greater_than_thresh(arr: Array, thresh: ArrayLike) -> ArrayLike:
    """Find the minimum value in an array that is greater than a threshold.

    Args:
        arr (Array): array to search.
        thresh (ArrayLike): threshold to compare against.

    Returns:
        ArrayLike: The minimum value in the array that is greater than the threshold.
    """
    # Mask the array to keep only values greater than the threshold.
    thresh_vals = jnp.where(arr > thresh, arr, jnp.inf)
    # Find the minimum value in the masked array
    return jnp.min(thresh_vals)


def contiguous_true_end_of_axis_mask(arr: np.ndarray, axis: int) -> np.ndarray:
    """Mask contiguous True values at the end of an array along a specified axis.
    This function is primarily for masking the part of a xr.DataArray with end-of-episode padding.

    Args:
        arr (np.ndarray): array of booleans to mask.
        axis (int): axis along which to mask contiguous True values.

    Returns:
        np.ndarray: Masked array.
    """
    # Reverse the array along the specified axis
    arr_rev = np.flip(arr, axis=axis)

    # Perform cumulative logical AND operation along the specified axis
    mask_rev = np.logical_and.accumulate(arr_rev, axis=axis)

    # Reverse the mask back to the original order
    mask = np.flip(mask_rev, axis=axis)

    return mask
