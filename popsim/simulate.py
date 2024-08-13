import typing
from enum import IntEnum

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, PyTree

import popsim.interp as pinterp
from popsim import ModuleBase
from popsim.hybrid_state import partition_discrete_cont
from popsim.interp import InterpType, resolve_paths
from popsim.param_utils import build_vectorized_params
from popsim.xarray_utils import solution_to_xarray, time_and_pytree_to_xarray


class StepperType(IntEnum):
    SIMPLE_EULER = 0
    DIFFRAX = 1


def generate_prng_trajectory(prng_key_seed: jax.random.PRNGKey, n_samps: int, time_base: Array) -> diffrax.LinearInterpolation:
    """Generate PRNG trajectories for all of the simulations.

    Args:
        prng_key_seed (jax.random.PRNGKey): seed used to randomly generate the PRNG trajectories.
        n_samps (int): number of simulations.
        time_base (Array): time base for the simulation.

    Returns:
        diffrax.LinearInterpolation: PRNG trajectories.
    """
    # If we have a seed, use it to generate a PRNG key for each simulation case and each time step.
    prng_key = jax.random.randint(
        prng_key_seed, (n_samps, time_base.size), minval=jnp.iinfo(jnp.int32).min, maxval=jnp.iinfo(jnp.int32).max
    )

    def interp_one_sim(key_for_one_sim):
        return pinterp.interp(time_base, key_for_one_sim, interp_type=pinterp.InterpType.LINEAR)

    prng_traj = jax.vmap(interp_one_sim)(prng_key)

    return prng_traj


def simulate(
    module: ModuleBase,
    time_base: Array,
    initial_state: PyTree,
    params: typing.Union[typing.Sequence[PyTree], PyTree],
    interp_type: InterpType = InterpType.LINEAR,
    return_xarray: bool = True,
    stepper_type: StepperType = StepperType.SIMPLE_EULER,
    prng_key_seed: typing.Optional[jax.random.PRNGKey] = None,
) -> typing.Union[diffrax.Solution, xr.Dataset]:
    """Simulate a module.
    Args:
        module (ModuleBase): the dynamics module to simulate.
        time_base (Array): the time base for the simulation.
        initial_state (PyTree): the initial state of the system. Should be DynamicsModule.State.
        params (typing.Union[typing.Sequence[PyTree], PyTree]): the parameters of the system. Should be DynamicsModule.Params or a sequence of DynamicsModule.Params.
        interp_type (InterpType, optional): interpolation method for params over time.
        return_xarray (bool, optional): whether to return a xr.Dataset or a diffrax.Solution. Defaults to True.
        stepper_type (StepperType, optional): stepper type to use. Defaults to StepperType.SIMPLE_EULER.
        prng_key_seed (typing.Optional[jax.random.PRNGKey], optional): If a random number key is provided, then PRNGKey trajectories will be auto-generated for the module for all simulation cases. If provided value is 'None', then 'None' will be passed to all modules as the 'key' argument. Defaults to None.
    Returns:
        typing.Union[diffrax.Solution, xr.Dataset]: simulation results.
    """
    if not isinstance(params, typing.Sequence):
        params = [params]

    # If we are using Diffrax, then we expect the initial state to be completely continuous.
    if stepper_type == StepperType.DIFFRAX:
        discrete_state, _ = partition_discrete_cont(initial_state)
        discrete_leaves = jax.tree.leaves(discrete_state)

        if len(discrete_leaves) > 0:
            raise ValueError("State must be completely continuous when using Diffrax for the simulation.")

    # Build the params.
    params_vectorized, nsims = build_vectorized_params(params, time_base, interp_type)

    multi_sim = nsims > 1

    # Choose the simulation function based on the stepper type.
    if stepper_type == StepperType.SIMPLE_EULER:
        simulate_fun = _simple_euler_simulate
    elif stepper_type == StepperType.DIFFRAX:
        simulate_fun = _diffrax_simulate
    else:
        raise ValueError("Stepper type not recognized.")

    # Perform the simulation.
    sol = (
        _vec_simulate(module, time_base, initial_state, params_vectorized, simulate_fun=simulate_fun)
        if multi_sim
        else simulate_fun(module, time_base, initial_state, params_vectorized)
    )

    if stepper_type == StepperType.SIMPLE_EULER:
        return time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim) if return_xarray else sol
    elif stepper_type == StepperType.DIFFRAX:
        return solution_to_xarray(sol, multi_simulation=multi_sim) if return_xarray else sol
    else:
        raise ValueError("Stepper type not recognized.")


@eqx.filter_jit
def _vec_simulate(module, ts, state0, params_vectorized, simulate_fun):
    params_axes = jax.tree.map(lambda _: 0, params_vectorized)
    # Perform a vectorized simulation.
    sol = jax.vmap(
        simulate_fun,
        in_axes=(None, None, None, params_axes),
    )(module, ts, state0, params_vectorized)

    # Remove extraneous dimensions.
    sol = jax.tree.map(lambda x: jnp.squeeze(x), sol)
    return sol


@eqx.filter_jit
def _diffrax_simulate(module: ModuleBase, ts: Array, state0, params) -> diffrax.Solution:
    def module_f(t, y, params, return_aux=False):
        params_resolved = resolve_paths(params, t)
        state_dot, out = module(y, params_resolved)
        if return_aux:
            return out, params_resolved
        else:
            return state_dot

    # Function to save auxiliary information.
    def saveat_fn(t, y, args):
        output, params_resolved = module_f(t, y, args, return_aux=True)
        out = {"state": y, "output": output, "params": params_resolved}
        return out

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(module_f),
        solver=diffrax.Euler(),
        t0=ts[0],
        t1=ts[-1],
        dt0=jnp.min(jnp.diff(ts)),
        y0=state0,
        args=params,
        saveat=diffrax.SaveAt(ts=ts, fn=saveat_fn),
        max_steps=None,  # Allows indefinite number of steps.
    )
    return sol


@eqx.filter_jit
def _simple_euler_simulate(module: ModuleBase, ts: Array, state0, params):
    dts = jnp.diff(ts)
    # Check dts are all equal.
    dts = eqx.error_if(dts, jnp.any(jnp.abs(dts - dts[0]) > 1e-10), "Time steps must be uniform.")

    dt = dts[0]

    def _euler_step(carry, t):
        state = carry
        params_resolved = resolve_paths(params, t)
        state_out, out = module(state, params_resolved)

        # Partition the state output tree into continuous (float, complex, and arrays of float + complex) and discrete parts (everything else).
        # The continuous parts are assumed to be state_dot. The discrete parts are assumed to be the next state.
        discrete_state_next, state_dot = partition_discrete_cont(state_out)

        def step_fn(x, xdot):
            return x + xdot * dt if xdot is not None else None

        # Perform an Euler step on the continuous part
        continuous_state_next = jax.tree.map(step_fn, state, state_dot)

        # Combine the next continuous state with the next discrete state
        state_next = eqx.combine(discrete_state_next, continuous_state_next)

        output_data = {
            "output": out,
            "state": state,
            "params": params_resolved,
        }

        return state_next, output_data

    _, outputs = jax.lax.scan(_euler_step, state0, xs=ts)
    return outputs
