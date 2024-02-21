import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from popsim.simulators.comet_mirror.model import CometMirror
from popsim.tree_util import resolve_paths


class Integrator(eqx.Module):
    model: CometMirror
    term: diffrax.ODETerm

    def __init__(self, model: CometMirror):
        self.model = model

        def model_f(t, y, params):
            params_resolved = resolve_paths(params, t)
            out = model(y, params_resolved, debug_info=False)
            return out

        self.term = diffrax.ODETerm(model_f)

    @eqx.filter_jit
    def __call__(self, ts, state0, params, debug_info=False):
        sol = diffrax.diffeqsolve(
            terms=self.term,
            solver=diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=jnp.min(jnp.diff(ts)),
            y0=state0,
            args=params,
            saveat=diffrax.SaveAt(ts=ts),
        )
        if debug_info:
            debugs = jax.vmap(lambda y, t: self.model(y, resolve_paths(params, t), debug_info))(sol.ys, sol.ts)
            return sol, debugs
        else:
            return sol
