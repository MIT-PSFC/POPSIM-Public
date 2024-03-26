import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array

import popsim.simulators.comet_mirror.model as cm
from popsim.tree_util import resolve_paths


def simulate(model: cm.CometMirror, ts: Array, state0: cm.State, params: cm.Params, debug_info=False) -> diffrax.Solution:
    def model_f(t, y, params):
        params_resolved = resolve_paths(params, t)
        out = model(y, params_resolved, debug_info=False)
        return out

    term = diffrax.ODETerm(model_f)
    sol = diffrax.diffeqsolve(
        terms=term,
        solver=diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=jnp.min(jnp.diff(ts)),
        y0=state0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts),
    )
    if debug_info:
        debugs = jax.vmap(lambda y, t: model(y, resolve_paths(params, t), debug_info))(sol.ys, sol.ts)
        return sol, debugs
    else:
        return sol
