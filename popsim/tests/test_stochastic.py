import diffrax
import jax
import jax.numpy as jnp

from popsim.enums import Impurity
from popsim.stochastic import generate_random_walks


def test_generate_random_walks():
    y0 = {
        Impurity.Helium: 7.0,
        Impurity.Beryllium: 7.0,
    }

    n_samples = 10
    diffusion_mags = {k: jnp.array(0.1) for k in y0.keys()}
    ts = jax.numpy.linspace(0, 1, 100)

    # Example function to generate shape and dtype structure of y0

    res = generate_random_walks(jax.random.PRNGKey(0), n_samples, ts, y0, diffusion_mags)

    assert res.ys[Impurity.Helium].shape == (n_samples, len(ts))
    assert res.ys[Impurity.Beryllium].shape == (n_samples, len(ts))