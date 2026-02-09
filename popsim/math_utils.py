import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike
from scipy.linalg import eig


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


def soft_clip(x: ArrayLike, min_value: ArrayLike, max_value: ArrayLike, sharpness: float = 2.0, eps: float = 1e-6) -> ArrayLike:
    """A smooth alternative to clipping that uses a tanh function.

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


def padded_relative_error(predicted: ArrayLike, target: ArrayLike, pad: float = 1.0) -> ArrayLike:
    """Relative error with a denominator padding term to avoid division by zero.

    Computes ``norm(predicted - target) / (norm(target) + pad)``.
    For scalar inputs this reduces to ``|predicted - target| / (|target| + pad)``.

    Args:
        predicted (ArrayLike): Predicted values.
        target (ArrayLike): Target (ground-truth) values.
        pad (float, optional): Additive constant in the denominator. Defaults to 1.0.

    Returns:
        ArrayLike: Scalar relative error.
    """
    return jnp.linalg.norm(predicted - target) / (jnp.linalg.norm(target) + pad)


def eigen_decompose(A: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose a square matrix, returning sorted eigenvalues and
    sign-normalised eigenvectors.

    Eigenvalues are sorted in descending order.  The sign of each eigenvector
    is chosen so that its first element is positive.  If *A* contains any NaN
    the function short-circuits and returns all-NaN arrays of the appropriate
    shape.

    Args:
        A: Square matrix to decompose.

    Returns:
        eigenvalues: 1-D array of eigenvalues sorted in descending order.
        V: 2-D array whose columns are the corresponding eigenvectors.
    """
    if np.isnan(A).any():
        n = A.shape[0]
        return np.full(n, np.nan), np.full((n, n), np.nan)

    eigenvalues, V = eig(A)

    # Convert to plain numpy (no-op if already numpy; handles JAX arrays).
    eigenvalues = np.asarray(eigenvalues)
    V = np.asarray(V)

    # Sort eigenvalues and V in descending order.
    sorted_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    V = V[:, sorted_indices]

    # Choose the sign of each eigenvector so that its first element is positive.
    signs = np.where(V[0, :] < 0, -1, 1)
    V = V * signs[np.newaxis, :]

    return eigenvalues, V
