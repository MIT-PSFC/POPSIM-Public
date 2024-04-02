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

    assert len(res) == n_samples
    for r in res:
        assert r.ys[Impurity.Helium].shape == (len(ts),)
        assert r.ys[Impurity.Beryllium].shape == (len(ts),)
