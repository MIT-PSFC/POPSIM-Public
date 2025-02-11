import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array

from popsim.utils import time_epsilon


def pad_time(times: Array) -> Array:
    """Diffrax has issues with both repeated times and nans in the time array.
    The solution is to repeat the last time but with a small epsilon added to it.
    This epsilon addition is handled by jnp.nextafter, which accounts for properly scaling the epsilon
    to yield a number which is actually greater than the input given floating point accuracy.

    Examples:
        jnp.array([0, 1, 2, 3, 4, 4, 4]) -> jnp.array([0, 1, 2, 3, 4, 4 + eps, 4 + 2 * eps])
        jnp.array([0, 1, nan, 3, 4, 4, 5]) -> jnp.array([0, 1, 1 + eps, 3, 4, 4 + eps, 5])

    Args:
        times (Array): array of times in ascending order possibly with repeat elements or nans.
    Returns:
        Array: Given an array of times that is sorted in ascending order and may contain repeat elements or nan elements in arbitrary locations,
        pad later elements with a small epsilon to ensure the times are strictly increasing.
    """

    def scan_func(carry, current_val):
        prev_val, _ = carry
        current_val = jax.lax.cond(
            jnp.greater(current_val, prev_val), lambda _: current_val, lambda _: prev_val + time_epsilon(prev_val), operand=None
        )
        return (current_val, _), current_val

    _, times = jax.lax.scan(scan_func, (times[0], -1), times)
    return times


def pad_time_xr(time: xr.DataArray, time_dim: str) -> xr.DataArray:
    """Given an xarray DataArray of times corresponding to episodes, apply time padding. This replaces nans and repeated times with strictly increasing times.

    Args:
        time (xr.DataArray): xarray DataArray of times.
        time_dim (str): name of the time dimension.

    Returns:
        xr.DataArray: the time DataArray with nans and repeated times replaced with strictly increasing times.
    """

    def _pad(t):
        # By default, xr.apply_ufunc moves core dimensions to the end of the array.
        # Thus, using np.apply_along_axis along the last dimension ensures "pad_time" is applied to the time dimension.
        return jnp.apply_along_axis(pad_time, -1, t)

    return xr.apply_ufunc(_pad, time, input_core_dims=[[time_dim]], output_core_dims=[[time_dim]])
