from popsim.modules.profile_predictor.module import ProfileShape
from popsim.modules.profile_predictor.train_configs import SPARC_CONFIG, TCV_CONFIG
from popsim.modules.profile_predictor.train import train
from popsim.basis import BSplineBasis, InterpedLinearBasis
import jax.numpy as jnp
import pytest
import equinox as eqx

@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("bspline", [True, False])
def test_profile_shape(normalize, bspline):
    coeffs = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0])
    rho = jnp.linspace(0, 1, 5)

    # Test the different ways to make a profile shape
    if bspline:
        profile_shape = ProfileShape.make_bspline(coeffs, normalize=normalize)
        assert isinstance(profile_shape.basis, BSplineBasis)
    else:
        profile_shape = ProfileShape.make_points(coeffs, rho, normalize=normalize)
        assert isinstance(profile_shape.basis, InterpedLinearBasis)

    # Check that the coeffs are interpolated correctly
    interp_eval = profile_shape(rho)

    if not normalize:
        assert jnp.array_equal(interp_eval, coeffs)
    else:
        assert jnp.allclose(interp_eval, coeffs / profile_shape.integral())

    # Check that an out of bounds rho raises an error
    with pytest.raises(eqx.EquinoxRuntimeError):
        profile_shape(jnp.array([-1.0]))

def test_train():
    config = SPARC_CONFIG.copy()
    config["max_epochs"] = 10
    config["epochs_per_val"] = 5
    config["debug"] = True
    train(config)