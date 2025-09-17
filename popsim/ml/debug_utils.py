import equinox as eqx
import jax
import jax.numpy as jnp
import loguru

from popsim.ml.dataloading import XarrayPreppedDataset
from popsim.ml.envs import ModuleTrainingEnv
from popsim.tree_util import keypath_to_string


def check_large_model_weights(model, threshold: float | None = 1e3) -> list[str]:
    """
    Check if any model weights exceed a certain threshold.
    Prints list of such weights and the corresponding parameter names as logger warnings

    Args:
        model: An Equinox model (or any pytree of arrays)
        threshold (Optional[float]): The threshold above which a weight is considered "large".

    Returns:
        List of paths (as strings) to weights with large values.
    """

    def _large_weights(leaf):
        return eqx.is_array(leaf) and jnp.any(jnp.abs(leaf) > threshold)

    large_weights_tree = eqx.filter(model, _large_weights)
    large_weights_leaves_with_paths = jax.tree_util.tree_leaves_with_path(large_weights_tree)

    error_paths = []
    for path, leaf in large_weights_leaves_with_paths:
        path_str = keypath_to_string(path)
        max_val = jnp.max(jnp.abs(leaf))
        error_paths.append(path_str)
        loguru.logger.warning(f"Large weight in {path_str}: max |weight| = {max_val}")

    return error_paths


def check_zero_variance_variables(data) -> list[str]:
    """
    Check if any variables in the data have zero variance.
    Prints list of such variables as logger warnings

    Args:
        data: A pytree of arrays (e.g., model parameters or batch of data)

    Returns:
        List of paths (as strings) to variables with zero variance.
    """

    def _zero_variance(leaf):
        if jnp.isscalar(leaf) or leaf.ndim == 0:
            return False  # Skip scalars
        else:
            return jnp.var(leaf) == 0

    zero_variance_tree = eqx.filter(data, _zero_variance)
    zero_variance_leaves_with_paths = jax.tree_util.tree_leaves_with_path(zero_variance_tree)

    error_paths = []
    for path, _ in zero_variance_leaves_with_paths:
        path_str = keypath_to_string(path)
        error_paths.append(path_str)
        loguru.logger.warning(f"Zero variance in {path_str}")

    return error_paths


def check_large_values(data, threshold: float | None = 1e3) -> list[str]:
    """
    Check if any values in the data exceed a certain threshold.
    Prints list of such variables as logger warnings

    Args:
        data: A pytree of arrays (e.g., model parameters or batch of data)
        threshold (Optional[float]): The threshold above which a value is considered "large".

    Returns:
        List of paths (as strings) to variables with large values.
    """

    def _large_values(leaf):
        if jnp.isscalar(leaf) or leaf.ndim == 0:
            return jnp.abs(leaf) > threshold
        else:
            return jnp.any(jnp.abs(leaf) > threshold)

    large_values_tree = eqx.filter(data, _large_values)
    large_values_leaves_with_paths = jax.tree_util.tree_leaves_with_path(large_values_tree)

    error_paths = []
    for path, leaf in large_values_leaves_with_paths:
        path_str = keypath_to_string(path)
        error_paths.append(path_str)
        if jnp.isscalar(leaf) or leaf.ndim == 0:
            loguru.logger.warning(f"Large value in {path_str}: {leaf}")
        else:
            max_val = jnp.max(jnp.abs(leaf))
            loguru.logger.warning(f"Large value in {path_str}: max |value| = {max_val}")

    return error_paths


def diagnose_nans(prev_model: ModuleTrainingEnv | None = None, batch: XarrayPreppedDataset | None = None):
    """
    Utility function for helping figure out why NaNs are appearing during training.

    Args:
        prev_model (Optional[ModuleTrainingEnv]): Previous model state before NaNs appeared, if available.
        batch (Optional[XarrayPreppedDataset]): Batch of data that caused NaNs, if available.
    """

    if prev_model is not None:
        loguru.logger.info("Diagnosing previous model...")
        check_large_model_weights(prev_model)

    if batch is not None:
        loguru.logger.info("Diagnosing batch...")
        inputs, targets = batch.get_inputs_and_targets()

        for name, data in zip(["inputs", "targets"], [inputs, targets], strict=False):
            loguru.logger.info(f"Checking {name} for zero variance...")
            check_zero_variance_variables(data)
            loguru.logger.info(f"Checking {name} for large values...")
            check_large_values(data)
