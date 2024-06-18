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
from popsim.xarray_utils import solution_to_xarray


def simulate(
    model: ModuleBase,
    time_base: Array,
    initial_state: PyTree,
    params: typing.Union[typing.Sequence[PyTree], PyTree],
    interp_type: str = "linear",
    return_xarray: bool = True,
) -> typing.Union[diffrax.Solution, xr.Dataset]:
    """Simulate a module.
    Args:
        model (ModuleBase): the dynamics module to simulate.
        time_base (Array): the time base for the simulation.
        initial_state (PyTree): the initial state of the system. Should be DynamicsModule.State.
        params (typing.Union[typing.Sequence[PyTree], PyTree]): the parameters of the system. Should be DynamicsModule.Params or a sequence of DynamicsModule.Params.
        interp_type (str, optional): interpolation method for params over time. Defaults to "linear".
        return_xarray (bool, optional): whether to return a xr.Dataset or a diffrax.Solution. Defaults to True.
    Returns:
        typing.Union[diffrax.Solution, xr.Dataset]: simulation results.
    """
    if not isinstance(params, typing.Sequence):
        params = [params]

    # Build the params.
    params_vectorized, multi_sim = build_vectorized_params(params, time_base, interp_type)

    # Perform the simulation.
    sol = _vec_simulate(model, time_base, initial_state, params_vectorized)

    sol = jax.tree_map(lambda x: jnp.squeeze(x), sol)

    return solution_to_xarray(sol, multi_simulation=multi_sim) if return_xarray else sol


@eqx.filter_jit
def _vec_simulate(model, ts, state0, params_vectorized):
    params_axes = jax.tree_map(lambda x: 0, params_vectorized)

    # Perform a vectorized simulation.
    sol = jax.vmap(
        _simulate,
        in_axes=(None, None, None, params_axes),
    )(model, ts, state0, params_vectorized)
    return sol


@eqx.filter_jit
def _simulate(model, ts: Array, state0, params) -> diffrax.Solution:
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
        max_steps=100000,
    )
    return sol


@eqx.filter_jit
def euler_multi_step(nmax, model, t0, dt, state0, params):
    def _euler_step(n, carry):
        state, t = carry
        params_resolved = resolve_paths(params, t)
        t = t + dt
        state_dot, out = model(state, params_resolved)
        return (jax.tree.map(lambda x, y: x + y * dt, state, state_dot), t)

    state, t = jax.lax.fori_loop(0, nmax, _euler_step, (state0, t0))
    return state, t
