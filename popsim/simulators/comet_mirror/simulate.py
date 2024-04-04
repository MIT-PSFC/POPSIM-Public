import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import popsim.simulators.comet_mirror.model as cm
from popsim.config import build_vectorized_configs
from popsim.interp import resolve_paths
from popsim.xarray_utils import solution_to_xarray


def simulate(
    model: cm.CometMirror,
    time_base: Array,
    initial_state: cm.State,
    params: typing.Union[cm.Params, typing.Sequence[cm.Params]],
    interp_type: str = "linear",
    return_xarray: bool = True,
) -> diffrax.Solution:
    if isinstance(params, cm.Params):
        params = [params]
    elif isinstance(params, typing.Sequence):
        pass
    else:
        raise ValueError("params must be either a single Params instance or a sequence of Params instances.")

    # Build the configurations.
    params_vectorized, multi_sim = build_vectorized_configs(params, time_base, interp_type)

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
def _simulate(model: cm.CometMirror, ts: Array, state0: cm.State, params: cm.Params) -> diffrax.Solution:
    def model_f(t, y, params, return_aux: bool = False):
        params_resolved = resolve_paths(params, t)
        out = model(y, params_resolved, return_aux=return_aux)
        return out

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
    )
    return sol
