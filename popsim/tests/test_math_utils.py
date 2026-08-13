import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from popsim.math_utils import inverse_signed_log, safe_log, signed_log, soft_clip


def test_normal_log_nan_issue():
    # Test values including zero and negative numbers
    x_values = jnp.array([-1.0, 0.0, 1.0, 2.0])

    def unsafe_fn(x):
        return jnp.where(x > 0, jnp.log(x), 0)
    
    def safe_fn(x):
        return jnp.where(x > 0, safe_log(x), 0)

    # Check gradient computation for NaNs
    grad_fn_unsafe = jax.grad(lambda x: jnp.sum(unsafe_fn(x)))

    grad_fn_safe = jax.grad(lambda x: jnp.sum(safe_fn(x)))
    
    grads_unsafe = grad_fn_unsafe(x_values)

    # Check there are nans present.
    assert jnp.any(jnp.isnan(grads_unsafe))

    # Now test that this issue is not present in the safe function.
    grads_safe = grad_fn_safe(x_values)
    assert not jnp.any(jnp.isnan(grads_safe))

def test_round_trip_signed_log():
    key = jax.random.PRNGKey(42)

    # Random samples
    n_samps = 1000
    random_samples_x = jax.random.uniform(key, (n_samps,), minval=-1000, maxval=1000)
    key, subkey = jax.random.split(key)
    random_samples_y = jax.random.uniform(subkey, (n_samps,), minval=-10, maxval=10)


    # Test signed_log then inverse_signed_log
    y = signed_log(random_samples_x)
    x_reconstructed = inverse_signed_log(y)
    assert jnp.allclose(random_samples_x, x_reconstructed, atol=1e-6)

    # Test inverse_signed_log then signed_log
    x = inverse_signed_log(random_samples_y)
    y_reconstructed = signed_log(x)
    assert jnp.allclose(random_samples_y, y_reconstructed, atol=1e-6)

    # Test that back-propagation works
    def fn(x):
        return jnp.sum(signed_log(x))
    
    grad_fn = jax.grad(fn)
    grads = grad_fn(random_samples_x)
    assert not jnp.any(jnp.isnan(grads))

    def fn_inv(y):
        return jnp.sum(inverse_signed_log(y))
    grad_fn_inv = jax.grad(fn_inv)
    grads_inv = grad_fn_inv(random_samples_y)
    assert not jnp.any(jnp.isnan(grads_inv))

@pytest.mark.parametrize("min_value, max_value, sharpness", [
    (-1.0, 1.0, 10.0),
    (0.0, 5.0, 5.0),
    (-2.0, 3.0, 20.0),
    (1e-5, 2e-5, 1.0),
])
def test_soft_clip(min_value: float, max_value: float, sharpness: float):
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (100,), minval=min_value, maxval=max_value)

    # Clip the values softly
    clipped_x = soft_clip(x, min_value, max_value, sharpness)

    # Check that clipped values are within bounds.
    assert jnp.all(clipped_x >= min_value)
    assert jnp.all(clipped_x <= max_value)
    
    # Test that back-propagation works
    def fn(x):
        return jnp.sum(soft_clip(x, min_value, max_value, sharpness))
    grads = jax.grad(fn)(x)
    assert not jnp.any(jnp.isnan(grads))


def test_soft_clip_vec():
    values = jnp.array([0.0, 0.0, 1.0])
    mins = jnp.array([-1.0, 0.0, 0.0])
    maxs = jnp.array([1.0, 1.0, 1.0])
    out = soft_clip(values, mins, maxs)
    assert out[0] == 0.0
    assert out[1] > 0.0 and out[1] < 1.0

    # Test that back-propagation works
    def fn(x):
        return jnp.sum(soft_clip(x, mins, maxs))
    grads = jax.grad(fn)(values)
    assert not jnp.any(jnp.isnan(grads))
