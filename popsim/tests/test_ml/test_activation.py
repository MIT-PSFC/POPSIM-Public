from popsim.ml.activation import Activation
import jax
import jax.numpy as jnp
import pytest

@pytest.mark.parametrize("activation", list(Activation))
def test_activation_gradients(activation):
    fn = activation.get_fn()
    jac_fn = activation.get_jac_fn()

    key = jax.random.PRNGKey(42)
    
    n_samps = 100000
    n_dim = 10
    x = jax.random.normal(key, (n_samps, n_dim))  # Generate test inputs

    # Compute analytical and autodiff gradients
    autodiff_grad = jax.vmap(jax.jacfwd(fn), in_axes=(0))(x)
    
    analytical_grad = jax.vmap(jac_fn)(x)

    # Check if analytical and autodiff gradients are close
    assert jnp.allclose(analytical_grad, autodiff_grad), f"Mismatch in {activation.value}"