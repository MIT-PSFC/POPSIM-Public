import diffrax
import jax.numpy as jnp
from jaxtyping import Array

import popsim.simulators.comet_mirror.model as cm
from popsim.tree_util import resolve_paths


def simulate(model: cm.CometMirror, ts: Array, state0: cm.State, params: cm.Params) -> diffrax.Solution:
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
