from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, PyTree


class ScalingType(StrEnum):
    """Enum for different scaling types used in data normalization."""

    STD = "std"
    MIN_MAX = "min_max"
    QUANTILE_50 = "quantile_50"
    L2 = "l2"
    NONE = "none"


def norm_data(tree: PyTree, scaling_type: ScalingType | str = ScalingType.STD, sample_dim: int = 0) -> tuple[PyTree, PyTree, PyTree]:
    """Normalize a PyTree of arrays by subtracting means and scaling.

    Args:
        tree: PyTree of arrays to normalize. Each array must be at most 2D.
        scaling_type: Type of scaling to apply. Can be ScalingType enum or string ("std", "l2", "none").
        sample_dim: Dimension along which to compute statistics (default 0).

    Returns:
        Tuple of (normalized_tree, means, scaling_factors).

    Raises:
        ValueError: If any array has more than 2 dimensions or if scaling_type is invalid.
    """

    # xr.Dataset is a special case. Handle it separately.
    if isinstance(tree, xr.Dataset):
        return norm_data_xr(tree, scaling_type, sample_dim)

    # Convert string to enum if needed
    if isinstance(scaling_type, str):
        scaling_type = ScalingType(scaling_type)

    # Make sure all arrays are no more than 2D.
    def check_ndim(x):
        if x.ndim > 2:
            raise ValueError(f"Input arrays must be at most 2D. Found array with shape {x.shape}.")
        return x

    tree = jax.tree.map(check_ndim, tree)

    means = jax.tree.map(lambda x: x.mean(axis=sample_dim), tree)
    tree = jax.tree.map(lambda x, mean: x - mean, tree, means)

    def scaling_factor_fn(x):
        if scaling_type == ScalingType.STD:
            return jnp.asarray(x.std(axis=sample_dim))
        elif scaling_type == ScalingType.MIN_MAX:
            return jnp.max(x, axis=sample_dim) - jnp.min(x, axis=sample_dim)
        elif scaling_type == ScalingType.L2:
            return jnp.mean(jnp.linalg.norm(x, axis=1))
        elif scaling_type == ScalingType.NONE:
            return jnp.ones(x.shape[1])
        else:
            raise ValueError(f"Invalid scaling_type: {scaling_type}. Choose from {list(ScalingType)}.")

    scaling_factor = jax.tree.map(scaling_factor_fn, tree)

    # Set scaling factors close to zero to one.
    scaling_factor = jax.tree.map(lambda x: x.at[jnp.isclose(x, 0.0)].set(1.0), scaling_factor)

    normed_tree = jax.tree.map(lambda x, scale: x / scale, tree, scaling_factor)
    return normed_tree, means, scaling_factor


def norm_data_xr(
    ds: xr.Dataset, scaling_type: ScalingType | str = ScalingType.STD, sample_dim: str = "sample"
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Normalize a Dataset by subtracting means and scaling.

    Args:
        ds: Dataset to normalize. Each variable must be at most 2D.
        scaling_type: Type of scaling to apply. Can be ScalingType enum or string ("std", "l2", "none").
        sample_dim: Dimension along which to compute statistics (default "sample").

    Returns:
        Tuple of (normalized_ds, means_ds, scaling_factors_ds).

    Raises:
        ValueError: If any variable has more than 2 dimensions or if scaling_type is invalid.
    """
    # Convert string to enum if needed
    if isinstance(scaling_type, str):
        scaling_type = ScalingType(scaling_type)

    means = ds.mean(dim=sample_dim)
    ds = ds - means

    def scaling_factor_fn(x):
        if scaling_type == ScalingType.STD:
            return x.std(dim=sample_dim)
        elif scaling_type == ScalingType.MIN_MAX:
            return x.max(dim=sample_dim) - x.min(dim=sample_dim)
        elif scaling_type == ScalingType.QUANTILE_50:
            # Drop the scalar "quantile" coordinate: with xarray_jax's global
            # arithmetic_compat="override" option, q75 - q25 would otherwise silently
            # keep the left operand's conflicting coordinate on the result.
            q75 = x.quantile(0.75, dim=sample_dim).drop_vars("quantile")
            q25 = x.quantile(0.25, dim=sample_dim).drop_vars("quantile")
            return q75 - q25
        elif scaling_type == ScalingType.L2:
            # Take the L2 norm across all non-sample dimensions and average over samples.
            return np.mean(np.sqrt((x**2).sum(dim=set(x.dims) - {sample_dim})))
        elif scaling_type == ScalingType.NONE:
            return xr.ones_like(x)
        else:
            raise ValueError(f"Invalid scaling_type: {scaling_type}. Choose from {list(ScalingType)}.")

    scaling_factor = ds.map(scaling_factor_fn)

    # Set scaling factors close to zero to one.
    def replace_close_to_zero(x):
        return xr.where(np.isclose(x, 0.0), 1.0, x)

    scaling_factor = scaling_factor.map(replace_close_to_zero)

    normed_ds = ds / scaling_factor

    return normed_ds, means, scaling_factor


def apply_norm(x: Array, mean: Array, scale: Array) -> Array:
    """Apply normalization to an array.

    Args:
        x: Array to normalize.
        mean: Mean to subtract.
        scale: Scaling factor to divide by.

    Returns:
        Normalized array.
    """
    return (x - mean) / scale


def apply_unnorm(x_norm: Array, mean: Array, scale: Array) -> Array:
    """Apply un-normalization to an array.

    Args:
        x_norm: Normalized array.
        mean: Mean to add.
        scale: Scaling factor to multiply by.
    Returns:
        Un-normalized array.
    """
    return x_norm * scale + mean
