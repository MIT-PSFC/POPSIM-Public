import jax.numpy as jnp
from jaxtyping import ArrayLike


def safe_log(x: ArrayLike) -> ArrayLike:
    """Backpropagation-safe log function. See:
    https://docs.kidger.site/equinox/api/debug/

    Args:
        x (ArrayLike): Input value.

    Returns:
        ArrayLike: Log of the input value.
    """
    safe_x = jnp.where(x > 0.0, x, 1.0)
    y = jnp.where(x > 0.0, jnp.log(safe_x), 0.0)
    return y


def signed_log(x: ArrayLike) -> ArrayLike:
    """Signed log that can take in positive and negative values.
    Note that the log is shifted by 1 to avoid a singularity at 0.
    A plot of this function reveals that it is smooth for the entire real line.

    Args:
        x (ArrayLike): Input value.

    Returns:
        ArrayLike: Signed log of the input value.
    """
    return jnp.sign(x) * safe_log(jnp.abs(x) + 1.0)


def inverse_signed_log(y: ArrayLike) -> ArrayLike:
    """Inverse of the signed log function.

    Args:
        y (ArrayLike): Input value.

    Returns:
        ArrayLike: Inverse of the signed log function.
    """
    return jnp.sign(y) * (jnp.exp(jnp.abs(y)) - 1)


def soft_clip(x: ArrayLike, min_value: ArrayLike, max_value: ArrayLike, sharpness: float = 1.0, eps: float = 1e-6) -> ArrayLike:
    """Softly clips a value (or vector) between minimum and maximum values (or vectors).
    This function provides a smooth alternative to the clip function.
    TODO(allenw): thsi function can have numerical issues with backpropagation. Investigate further.

    Args:
        x (ArrayLike): the value(s) to be clipped.
        min_value (ArrayLike): the minimum value(s) to clip to.
        max_value (ArrayLike): the maximum value(s) to clip to.
        sharpness (float, optional): controls how hard the transition is. Infinite sharpness should correspond to hard clipping. Defaults to 1.0.

    Returns:
        ArrayLike: the clipped value(s).
    """
    # Compute the center and half the range.
    center = (min_value + max_value) / 2.0
    half_range = (max_value - min_value) / 2.0

    # Avoid division by a tiny half_range by replacing values smaller than eps with eps.
    effective_half_range = jnp.where(jnp.abs(half_range) < eps, eps, half_range)

    # Apply the soft clipping using a scaled hyperbolic tangent.
    return center + half_range * jnp.tanh(sharpness * (x - center) / effective_half_range)
