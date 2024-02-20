import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from popsim.simulators.comet_mirror.model import CometMirror


class Integrator(eqx.Module):
    model: CometMirror
    term: diffrax.ODETerm

    def __init__(self, model: CometMirror):
        self.model = model

        def model_f(t, y, args):
            out = model(y, *args)
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
            args=(params,),
            saveat=diffrax.SaveAt(ts=ts),
        )
        if debug_info:
            debugs = jax.vmap(lambda y: self.model(y, params, debug_info=True))(sol.ys)
            return sol, debugs
        else:
            return sol
