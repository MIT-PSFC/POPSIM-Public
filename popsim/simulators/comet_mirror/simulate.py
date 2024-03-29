import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import popsim.simulators.comet_mirror.model as cm
from popsim.tree_util import resolve_paths, tree_transpose
from popsim.xarray_utils import solution_to_xarray


def simulate(
    model: cm.CometMirror,
    ts: Array,
    state0: cm.State,
    params: typing.Union[cm.Params, typing.Sequence[cm.Params]],
    return_xarray: bool = False,
) -> diffrax.Solution:
    if isinstance(params, cm.Params):
        return solution_to_xarray(_simulate(model, ts, state0, params)) if return_xarray else _simulate(model, ts, state0, params)
    elif isinstance(params, typing.Sequence):
        # We first need to perform a tree-transpose to vectorize the parameters.
        params_vectorized = tree_transpose(params)
        params_axes = jax.tree_map(lambda x: 0, params_vectorized)

        # Perform a vectorized simulation.
        sol = jax.vmap(
            _simulate,
            in_axes=(params_axes,),
        )(params_vectorized)
        return solution_to_xarray(sol, multi_simulation=True) if return_xarray else sol
    else:
        raise ValueError("params must be either a single Params instance or a sequence of Params instances.")


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
