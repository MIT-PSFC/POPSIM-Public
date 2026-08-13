import jax.numpy as jnp

from popsim.modules.lti import LTI
from popsim.simulate import SimInput, simulate


def test_lti_identity_system():
    """
    Simple stable + fully observable system where A = -I, B = I, C = I, D = 0.
    """
    n = 3
    timebase = jnp.linspace(0.0, 100, 1000)
    A = -1.0 * jnp.eye(n)
    B = jnp.eye(n)
    C = jnp.eye(n)
    D = jnp.zeros((n, n))

    lti = LTI(LTI.Config(A=A, B=B, C=C, D=D))

    # Simulation one: zero inputs. Expect state to decay to near zero.
    x0 = jnp.array([1.0, 2.0, 3.0])
    sim_inputs = SimInput(
        time=timebase,
        initial_state=LTI.State(x=x0),
        inputs = LTI.Params(u=jnp.zeros(n))
    )
    out = simulate(lti, sim_inputs)

    last_time_step = out.isel(time=-1)
    assert jnp.max(jnp.abs(last_time_step["output.y"].values)) < 1e-5


    # Simulation two: start at origin, and with an input of "u" we expect the corresponding state to approach "u".
    x0 = jnp.zeros(n)
    u = jnp.array([0.5, -1.0, 2.0])
    sim_inputs = SimInput(
        time=timebase,
        initial_state=LTI.State(x=x0),
        inputs = LTI.Params(u=u)
    )
    out = simulate(lti, sim_inputs)
    last_time_step = out.isel(time=-1)
    assert jnp.allclose(last_time_step["state.x"].values, u, atol=1e-3)

def test_lti_reduced_system():
    """
    Test a system with dimension 4, but the last one doesn't do anything, so we can reduce it out.
    """
    n = 4
    n_effective = 3
    timebase = jnp.linspace(0.0, 100, 1000)
    A = jnp.array([[-1.0, 0.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0]])
    B = jnp.eye(n)
    C = jnp.eye(n)
    D = jnp.zeros((n, n))

    lti = LTI(LTI.Config(A=A, B=B, C=C, D=D))
    lti = lti.reduce_eigen(n_modes=n_effective, max_eigenvalue=0.0)
    assert lti.config.eigen_reduced_data.Lambda.shape == (n_effective, n_effective)


    # Run a simulation to test that only the first three states respond to inputs.
    x0 = jnp.zeros(n)
    x0 = x0.astype(jnp.complex128) # Need complex type for eigen-reduced LTI.
    u = jnp.array([0.5, -1.0, 2.0, 100.0]) # We expect the last input to have no effect.
    sim_inputs = SimInput(
        time=timebase,
        initial_state=LTI.State(x=x0),
        inputs = LTI.Params(u=u)
    )


    out = simulate(lti, sim_inputs)
    last_time_step = out.isel(time=-1)

    # First three states arrive at "u", last state remains near zero.
    assert jnp.allclose(last_time_step["state.x"].values[0:n_effective], u[0:n_effective], atol=1e-3)
    assert jnp.allclose(last_time_step["state.x"].values[n_effective:], jnp.zeros(n - n_effective), atol=1e-5)
