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


def _repeat_time_hack(times: Array) -> Array:
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
            jnp.greater(current_val, prev_val), lambda _: current_val, lambda _: jnp.nextafter(prev_val, jnp.inf), operand=None
        )
        return (current_val, _), current_val

    _, times = jax.lax.scan(scan_func, (times[0], -1), times)
    return times
