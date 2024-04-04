import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree


def generate_shapes_and_dtypes(y0: PyTree[ArrayLike]) -> PyTree[jax.ShapeDtypeStruct]:
    # Promote y0.
    y0 = jax.tree_map(jnp.asarray, y0)
    # Generate the shape and dtype structure of y0.
    return jax.tree_map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), y0)


def _generate_random_walks(key, n_samps, ts, y0, drift, diffusion):
    t0, t1 = ts[0], ts[-1]
    dt0 = jnp.min(jnp.diff(ts))

    # Function to generate a single random walk
    def generate_single_random_walk(key):
        brownian_motion = diffrax.UnsafeBrownianPath(shape=generate_shapes_and_dtypes(y0), key=key)
        terms = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))
        solver = diffrax.Euler()
        sol = diffrax.diffeqsolve(terms, solver, t0, t1, dt0=dt0, y0=y0, saveat=diffrax.SaveAt(ts=ts), adjoint=diffrax.DirectAdjoint())
        return sol

    # Generate a batch of PRNGKeys
    keys = jax.random.split(key, n_samps)

    # Use vmap to vectorize the random walk generation
    generate_random_walks_vmap = jax.vmap(generate_single_random_walk)
    solutions = generate_random_walks_vmap(keys)

    return solutions


def generate_random_walks(
    key: jax.random.PRNGKey,
    n_samps: int,
    ts: jnp.ndarray,
    y0: PyTree[ArrayLike],
    diffusion_mags: PyTree[ArrayLike],
):
    # Assert y0 and diffusion_mags have the same structure.
    assert jax.tree.structure(y0) == jax.tree.structure(diffusion_mags)

    drift_struct = jax.tree_map(lambda leaf: jnp.zeros_like(leaf), y0)

    def drift(t, y, args):
        return drift_struct

    def diffusion(t, y, args):
        return diffusion_mags

    sol = _generate_random_walks(key, n_samps, ts, y0, drift, diffusion)
    return sol


def interp_random_walk_solution(sol: diffrax.Solution):
    n_samps = sol.ts.shape[0]
    return [diffrax.LinearInterpolation(ts=sol.ts[i], ys=jax.tree_map(lambda x, i=i: x[i], sol.ys)) for i in range(n_samps)]
