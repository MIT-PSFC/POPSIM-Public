import jax
import jax.numpy as jnp
from jaxtyping import Array


def _count_repeat_elements(times: Array) -> Array:
    """Given an array of times that is sorted in ascending order and may contain repeat elements at
    the end, count the cumulative number of times an element is repeated.

    Example:
        jnp.array([0, 1, 2, 3, 4, 4, 4]) -> jnp.array([0, 0, 0, 0, 0, 1, 2])

    Args:
        times (Array): array of times in ascending order possibly with repeat elements at the end.

    Returns:
        Array: the cumulative number of times an element is repeated.
    """

    def scan_func(carry, current):
        prev_val, prev_count = carry
        diff = jax.lax.cond(jnp.equal(prev_val, current), lambda _: 1, lambda _: 0, operand=None)
        count = prev_count + diff
        return (current, count), count

    _, counts = jax.lax.scan(scan_func, (times[0], -1), times)
    return counts


def _repeat_time_hack(times: Array, eps_mult: int = 1) -> Array:
    """Diffrax has issues with both repeated times and nans in the time array. The solution is to repeat the last
    time but with a small epsilon added to it.

    Example:
        jnp.array([0, 1, 2, 3, 4, 4, 4]) -> jnp.array([0, 1, 2, 3, 4, 4 + eps_mult * eps, 4 + 2 * eps_mult * eps])

    Args:
        times (Array): array of times in ascending order possibly with repeat elements at the end.
    Returns:
        Array: Given an array of times that is sorted in ascending order and may contain repeat elements at
        the end, pad the repeated elements with a small epsilon.
    """
    repeat_counts = _count_repeat_elements(times)
    epsilon = jnp.finfo(times.dtype).eps  # Machine epsilon for the dtype of times.
    return times + eps_mult * epsilon * repeat_counts
