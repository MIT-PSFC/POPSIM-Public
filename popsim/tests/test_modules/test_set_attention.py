from popsim.modules.set_attention import SetAttentionBlock
import jax
import jax.numpy as jnp
import pytest


@pytest.mark.parametrize("layernorm", [True, False])
def test_set_attention_invariance(layernorm):
    module = SetAttentionBlock(
        dim_in=2,
        dim_out=3,
        num_heads=2,
        key=jax.random.PRNGKey(0),
        layernorm=layernorm,
    )
    
    sequence = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    out1 = module(sequence)
    
    permuted_sequence = jnp.array([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    out2 = module(permuted_sequence)
    
    assert jnp.allclose(jnp.sort(out1, axis=0), jnp.sort(out2, axis=0)), "SetAttentionBlock is not permutation invariant."