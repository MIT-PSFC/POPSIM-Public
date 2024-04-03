import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import popsim.simulators.comet_mirror.model as cm
from popsim.config_utils import build_configs
from popsim.interp import resolve_paths
from popsim.tree_util import tree_transpose
from popsim.xarray_utils import solution_to_xarray


def unpack_lists(inp):
    unpacked_list = []
    for item in inp:
        if isinstance(item, list):
            unpacked_list.extend(item)
        else:
            unpacked_list.append(item)
    return unpacked_list


def simulate(
    model: cm.CometMirror,
    ts: Array,
    state0: cm.State,
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

    # First build the parameter configurations.
    params = [build_configs(p, interp_type) for p in params]
    params = unpack_lists(params)
    multi_sim = len(params) > 1

    # We need to perform a tree-transpose to vectorize the parameters.
    params_vectorized = tree_transpose(params)

    # Perform the simulation.
    sol = _vec_simulate(model, ts, state0, params_vectorized)

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
        return y, model_f(t, y, args, return_aux=True)

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
