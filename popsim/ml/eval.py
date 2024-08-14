import diffrax
import equinox as eqx
import jax

from popsim import interp
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.types import EvalInput
from popsim.ml.utils import _repeat_time_hack
from popsim.simulate import _diffrax_simulate


@eqx.filter_jit
def eval_module(env: ModuleEvalEnv, inputs: EvalInput) -> diffrax.Solution:
    # Use the environment-specified functions to create the initial state and parameter structures.
    initial_state = env.create_state(data=inputs.state_init)
    params = env.create_params(data=inputs.params)

    # Interpolate the parameters with rectilinear interpolation to allow for finer time steps.
    params_interped = interp.interp(times=_repeat_time_hack(inputs.time), tree=params, interp_type=interp.InterpType.RECTILINEAR)

    # Simulate the model.
    sol = _diffrax_simulate(module=env.module, ts=inputs.time, state0=initial_state, params=params_interped, prng_traj=None)
    return sol


@eqx.filter_jit
def vec_eval_module(env: ModuleEvalEnv, inputs: EvalInput) -> diffrax.Solution:
    input_axes = jax.tree.map(lambda _: 0, inputs)
    return jax.vmap(eval_module, in_axes=(None, input_axes))(env, inputs)
