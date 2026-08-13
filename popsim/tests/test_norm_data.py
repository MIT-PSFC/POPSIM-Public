import chex
import jax
import jax.numpy as jnp
import pytest
import xarray as xr

from popsim.norm_data import ScalingType, apply_norm, apply_unnorm, norm_data, norm_data_xr
from popsim.tests.fixtures import scrambled_multishot_liuqe_dataset


@pytest.mark.parametrize("scaling_type", list(ScalingType))
def test_norm_data_xr(scrambled_multishot_liuqe_dataset, scaling_type):
    ds = scrambled_multishot_liuqe_dataset
    dss = ds.stack(sample=("shot", "time_idx"))
    dss_normed, means, scales = norm_data_xr(dss, scaling_type=scaling_type, sample_dim="sample")
    
    # Un-normalize manually and check we get back the original data.
    dss_check = dss_normed * scales + means
    xr.testing.assert_allclose(dss, dss_check)

    if scaling_type == ScalingType.NONE:
        xr.testing.assert_allclose(scales, xr.ones_like(scales))
        xr.testing.assert_allclose(dss_normed, dss - means)
        
def test_simple_norm():
    tree = {
        "a": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        "b": {
            "c": jnp.array([10.0, 20.0, 30.0]),
            "d": jnp.array([[100.0], [200.0], [300.0]]),
        },
    }
    
    normed_tree, means, scales = norm_data(tree, scaling_type=ScalingType.MIN_MAX)
    
    # Check that un-normalizing gets back the original data.
    unnormed_tree = jax.tree.map(apply_unnorm, normed_tree, means, scales)
    chex.assert_trees_all_close(tree, unnormed_tree)
    
    # Due to min-max scaling, all values should be in [-0.5, 0.5].
    assert normed_tree["a"].min() == -0.5
    assert normed_tree["a"].max() == 0.5
    assert normed_tree["b"]["c"].min() == -0.5
    assert normed_tree["b"]["c"].max() == 0.5
    assert normed_tree["b"]["d"].min() == -0.5
    assert normed_tree["b"]["d"].max() == 0.5
    
    # Check that apply_norm works.
    normed_tree2 = jax.tree.map(apply_norm, tree, means, scales)
    chex.assert_trees_all_close(normed_tree, normed_tree2)