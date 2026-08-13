import chex
import jax
import jax.numpy as jnp
import pytest

from popsim import TimeIndepModule
from popsim.modules.bounded_output import BoundedOutput, BoundType


@pytest.mark.parametrize("bound_type", [BoundType.SOFT, BoundType.CLIP])
def test_bounded_output_vec_output(bound_type):
    class DummyModuleVecOut(TimeIndepModule):
        """Simple test module that multiplies inputs by -1."""
        def __call__(self, inputs):
            return -1.0 * inputs


    module = BoundedOutput(
        module=DummyModuleVecOut(),
        lower_bound=jnp.array([-0.5, -2.0, -3.0, -3.0, -3.0]),
        upper_bound=jnp.array([0.5, 2.0, 3.0, 3.0, 3.0]),
        bound_type=bound_type
    )

    inputs = jnp.array([1.0, 1.0, -1.0, 5.0, -5.0])
    output = module(inputs)
    
    if bound_type == BoundType.SOFT:
        # Expect outputs are within bounds.
        assert jnp.all(output >= module.lower_bound)
        assert jnp.all(output <= module.upper_bound)
    elif bound_type == BoundType.CLIP:
        expected_output = jnp.array([-0.5, -1.0, 1.0, -3.0, 3.0])
        chex.assert_trees_all_close(output, expected_output)
    else:
        raise ValueError(f"Test not set up for bound_type: {bound_type}")

@pytest.mark.parametrize("bound_type", [BoundType.SOFT, BoundType.CLIP])
def test_bounded_output_pytree_output(bound_type):
    class DummyModulePyTreeOut(TimeIndepModule):
        """Test to make sure that BoundedOutput works with PyTree outputs."""
        def __call__(self, inputs):
            return {
                "a": -1.0 * inputs,
                "b": {
                    "c": 2.0 * inputs,
                    "d": -3.0 * inputs
                }
            }
        
        
    module = BoundedOutput(
        module=DummyModulePyTreeOut(),
        lower_bound={
            "a": -0.5,
            "b": {
                "c": -1.0,
                "d": -1.5
            }
        },
        upper_bound={
            "a": 0.5,
            "b": {
                "c": 1.0,
                "d": 1.5
            }
        },
        bound_type=bound_type
    )
    
    inputs = jnp.array([1.0, -1.0])
    
    output = module(inputs)
    
    if bound_type == BoundType.SOFT:
        # Check that all outputs are within bounds.
        def check_bounds(out, lb, ub):
            assert jnp.all(out >= lb)
            assert jnp.all(out <= ub)
        
        jax.tree.map(
            check_bounds,
            output,
            module.lower_bound,
            module.upper_bound
        )
    elif bound_type == BoundType.CLIP:
        expected_output = {
            "a": jnp.array([-0.5, 0.5]),
            "b": {
                "c": jnp.array([1.0, -1.0]),
                "d": jnp.array([-1.5, 1.5])
            }
        }
        chex.assert_trees_all_close(output, expected_output)
    else:
        raise ValueError(f"Test not set up for bound_type: {bound_type}")