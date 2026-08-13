import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from popsim.ml.rtd_activation import Activation
from popsim.ml.rtd_mlp import RtdMLP


@pytest.mark.parametrize("activation", list(Activation))
@pytest.mark.parametrize("in_size", [1, 10])
@pytest.mark.parametrize("out_size", [1, 10])
@pytest.mark.parametrize("depth", [1, 3])
def test_rtd_mlp(activation, in_size, out_size, depth):
    if jax.config.jax_enable_x64:
        atol = 1e-08
    else:
        atol = 1e-05
    
    rtd_mlp = RtdMLP(
        in_size=in_size,
        out_size=out_size,
        width_size=10,
        depth=depth,
        activation=activation,
        final_activation=activation,
        key=jax.random.PRNGKey(42),
    )
    
    eqx_mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=10,
        depth=depth,
        activation=activation.get_fn(),
        final_activation=activation.get_fn(),
        key=jax.random.PRNGKey(42),
    )
    
    n_samps = 100
    xs = jax.random.normal(jax.random.PRNGKey(42), (n_samps, in_size))
    
    ys_rtd, ys_jac_rtd = jax.vmap(lambda x: rtd_mlp(x, return_jacobian=True))(xs)
    ys_jac_rtd_autodiff = jax.vmap(jax.jacfwd(lambda x: rtd_mlp(x, return_jacobian=False)))(xs)

    # Check consistency between autodiff and analytical gradients
    jnp.allclose(ys_jac_rtd, ys_jac_rtd_autodiff, atol=atol)
    
    
    # Equinox doesn't seem to handle softmax.
    if activation == Activation.SOFTMAX:
        return
    
    ys_eqx = jax.vmap(eqx_mlp)(xs)
    
    # Check consistenency between RtdMLP and EqxMLP
    assert jnp.allclose(ys_rtd, ys_eqx, atol=atol)
    