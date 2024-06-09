import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, PyTree

from popsim import ModuleBase
from popsim.interp import resolve_paths
from popsim.param_utils import build_vectorized_params
from popsim.xarray_utils import solution_to_xarray, time_and_pytree_to_xarray


def simulate(
    model: ModuleBase,
    time_base: Array,
    initial_state: PyTree,
    params: typing.Union[typing.Sequence[PyTree], PyTree],
    interp_type: str = "linear",
    return_xarray: bool = True,
    use_simple_euler: bool = False,
) -> typing.Union[diffrax.Solution, xr.Dataset]:
    """Simulate a module.
    Args:
        model (ModuleBase): the dynamics module to simulate.
        time_base (Array): the time base for the simulation.
        initial_state (PyTree): the initial state of the system. Should be DynamicsModule.State.
        params (typing.Union[typing.Sequence[PyTree], PyTree]): the parameters of the system. Should be DynamicsModule.Params or a sequence of DynamicsModule.Params.
        interp_type (str, optional): interpolation method for params over time. Defaults to "linear".
        return_xarray (bool, optional): whether to return a xr.Dataset or a diffrax.Solution. Defaults to True.
        use_simple_euler (bool, optional): whether to use a simple Euler method as opposed to Diffrax for the simulation. Defaults to False.
    Returns:
        typing.Union[diffrax.Solution, xr.Dataset]: simulation results.
    """
    if not isinstance(params, typing.Sequence):
        params = [params]

    # Build the params.
    params_vectorized, multi_sim = build_vectorized_params(params, time_base, interp_type)

    # Perform the simulation.
    simulate_fun = euler_multi_step if use_simple_euler else _diffrax_simulate
    sol = _vec_simulate(model, time_base, initial_state, params_vectorized, simulate_fun=simulate_fun)

    sol = jax.tree_map(lambda x: jnp.squeeze(x), sol)

    if use_simple_euler:
        return time_and_pytree_to_xarray(sol, time_base, multi_simulation=multi_sim) if return_xarray else sol
    else:
        return solution_to_xarray(sol, multi_simulation=multi_sim) if return_xarray else sol


@eqx.filter_jit
def _vec_simulate(model, ts, state0, params_vectorized, simulate_fun):
    params_axes = jax.tree_map(lambda x: 0, params_vectorized)

    # Perform a vectorized simulation.
    sol = jax.vmap(
        simulate_fun,
        in_axes=(None, None, None, params_axes),
    )(model, ts, state0, params_vectorized)
    return sol


@eqx.filter_jit
def _diffrax_simulate(model, ts: Array, state0, params) -> diffrax.Solution:
    def model_f(t, y, params, return_aux=False):
        params_resolved = resolve_paths(params, t)
        state_dot, out = model(y, params_resolved)
        if return_aux:
            return out
        else:
            return state_dot

    # Function to save auxiliary information.
    def saveat_fn(t, y, args):
        out = {"state": y, "aux": model_f(t, y, args, return_aux=True)}
        return out

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(model_f),
        solver=diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=jnp.min(jnp.diff(ts)),
        y0=state0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts, fn=saveat_fn),
        max_steps=max_steps,
    )
    return sol


@eqx.filter_jit
def euler_multi_step(model, ts: Array, state0, params):
    dts = jnp.diff(ts)
    # Check dts are all equal.
    dts = eqx.error_if(dts, jnp.any(jnp.abs(dts - dts[0]) > 1e-10), "Time steps must be equal.")

    dt = dts[0]

    def _euler_step(carry, t):
        state = carry
        params_resolved = resolve_paths(params, t)
        state_out, out = model(state, params_resolved)

        # Partition the state output tree into continuous (float, complex, and arrays of float + complex) and discrete parts (everything else).
        # The continuous parts are assumed to be state_dot. The discrete parts are assumed to be the next state.
        state_dot, discrete_state_next = eqx.partition(state_out, eqx.is_inexact_array_like)

        def step_fn(x, xdot):
            return x + xdot * dt if xdot is not None else None

        # Perform an Euler step on the continuous part
        continuous_state_next = jax.tree_map(step_fn, state, state_dot)

        # Combine the next continuous state with the next discrete state
        state_next = eqx.combine(continuous_state_next, discrete_state_next)

        return state_next, state_next

    _, states = jax.lax.scan(_euler_step, state0, xs=ts)
    return states
