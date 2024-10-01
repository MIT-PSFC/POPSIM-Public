import typing
import warnings
from enum import IntEnum

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import PyTree
from loguru import logger

from popsim import ModuleBase, config
from popsim.array_utils import min_greater_than_thresh
from popsim.hybrid_state import partition_discrete_cont
from popsim.interp import InterpType, resolve_paths
from popsim.modules.prng import PRNGModule
from popsim.param_utils import param_specs_to_paths
from popsim.sim_utils import (
    CombinatorialCases,  # . Import is used to allow the user to import this function from this module.
    MultiCases,  # . Import is used to allow the user to import this function from this module.
    SimInput,
    make_time_base,  # noqa: F401. Import is used to allow the user to import this function from this module.
)
from popsim.tree_util import get_instances_from_tree_leaves, tree_transpose
from popsim.xarray_utils import (
    solution_to_xarray,
    time_and_pytree_to_xarray,
)


class StepperType(IntEnum):
    SIMPLE_EULER = 0
    DIFFRAX = 1


def _check_sim_inputs(module: ModuleBase, sim_inputs: typing.Sequence[SimInput], stepper_type: StepperType):
    """Perform checks of the simulation inputs."""

    # Check if the state has any discrete components when using Diffrax. If so, raise an error.
    if stepper_type == StepperType.DIFFRAX:

        def _has_discrete_state(sim_input):
            discrete_state, _ = partition_discrete_cont(sim_input.initial_state)
            discrete_leaves = jax.tree.leaves(discrete_state)
            return len(discrete_leaves) > 0

        has_discrete_state = any(_has_discrete_state(sim_input) for sim_input in sim_inputs)
        if has_discrete_state:
            raise ValueError("State must be completely continuous when using Diffrax for the simulation.")

    # If PRNGModule.State is in the state, check that the seed is unique for each simulation.
    # If not, issue a warning.
    seed_states = get_instances_from_tree_leaves(sim_inputs, PRNGModule.State)
    if len(seed_states) > 1:
        seeds = [state.seed for state in seed_states]
        if len(seeds) != len(set(seeds)):
            warnings.warn(f"Multiple simulations have the same PRNG seed. Simulation PRNG seeds: {seeds}.", stacklevel=2)

    # Make sure that sim_inputs do not have any instances of MultiCases or CombinatorialCases.
    multi_cases = get_instances_from_tree_leaves(sim_inputs, MultiCases)
    if multi_cases:
        raise ValueError(
            "MultiCases detected in simulation inputs. Recommend explicitly calling .generate_cases() on them before calling simulate."
        )

    combinatorial_cases = get_instances_from_tree_leaves(sim_inputs, CombinatorialCases)
    if combinatorial_cases:
        raise ValueError(
            "CombinatorialCases detected in simulation inputs. Recommend explicitly calling .generate_cases() on them before calling simulate."
        )


def simulate(
    module: ModuleBase,
    sim_inputs: typing.Union[SimInput, typing.Sequence[SimInput]],
    interp_type: InterpType = InterpType.LINEAR,
    return_xarray: bool = True,
    stepper_type: StepperType = StepperType.SIMPLE_EULER,
) -> typing.Union[diffrax.Solution, xr.Dataset]:
    """Public facing API for simulating a module.

    Args:
        module (ModuleBase): the dynamics module to simulate.
        sim_inputs (typing.Union[SimInput, typing.Sequence[SimInput]]): simulation input.
        interp_type (InterpType, optional): interpolation method for params over time.
        return_xarray (bool, optional): whether to return a xr.Dataset or a diffrax.Solution. Defaults to True.
        stepper_type (StepperType, optional): stepper type to use. Defaults to StepperType.SIMPLE_EULER.

    Returns:
        typing.Union[diffrax.Solution, xr.Dataset]: simulation results.
    """
    if not isinstance(sim_inputs, typing.Sequence):
        sim_inputs = [sim_inputs]

    _check_sim_inputs(module, sim_inputs, stepper_type)

    # Resolve the param specifications to paths.
    sim_inputs = param_specs_to_paths(sim_inputs=sim_inputs, interp_type=interp_type)

    # Choose the simulation function based on the stepper type.
    if stepper_type == StepperType.SIMPLE_EULER:
        simulate_fun = _simple_euler_simulate
    elif stepper_type == StepperType.DIFFRAX:
        simulate_fun = _diffrax_simulate
    else:
        raise ValueError("Stepper type not recognized.")

    # Vectorize the simulation inputs.
    sim_inputs_vectorized = tree_transpose(sim_inputs)

    multi_sim = len(sim_inputs) > 1
    logger.info(f"Running {len(sim_inputs)} simulations.")

    # Perform the simulation.
    if multi_sim:
        sol = _vec_simulate(module, sim_inputs_vectorized, simulate_fun=simulate_fun)
    else:
        sol = simulate_fun(module, sim_inputs_vectorized)

    if stepper_type == StepperType.SIMPLE_EULER:
        return time_and_pytree_to_xarray(sim_inputs_vectorized.time, sol, multi_simulation=multi_sim) if return_xarray else sol
    elif stepper_type == StepperType.DIFFRAX:
        return solution_to_xarray(sol, multi_simulation=multi_sim) if return_xarray else sol
    else:
        raise ValueError("Stepper type not recognized.")


@eqx.filter_jit
def _vec_simulate(module: ModuleBase, sim_input: SimInput, simulate_fun):
    """Perform a vectorized simulation."""
    sim_input_axes = jax.tree.map(lambda x: 0, sim_input)

    sol = jax.vmap(
        simulate_fun,
        in_axes=(None, sim_input_axes),
    )(module, sim_input)

    # Remove extraneous dimensions.
    sol = jax.tree.map(lambda x: jnp.squeeze(x), sol)
    return sol


@eqx.filter_jit
def _diffrax_simulate(module: ModuleBase, sim_input: SimInput) -> diffrax.Solution:
    """Function for simulating a single case using diffrax."""

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

    # Get the minimum time step that is greater than zero.
    dt0 = min_greater_than_thresh(jnp.diff(sim_input.time), 0.0)

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(module_f),
        solver=diffrax.Euler(),
        t0=sim_input.time[0],
        t1=sim_input.time[-1],
        dt0=dt0,
        y0=sim_input.initial_state,
        args=sim_input.params,
        saveat=diffrax.SaveAt(ts=sim_input.time, fn=saveat_fn),
        max_steps=config["DIFFRAX_MAX_STEPS"],
    )
    return sol


@eqx.filter_jit
def _simple_euler_simulate(module: ModuleBase, sim_input: SimInput) -> PyTree:
    """Function for simulating a single case using simple Euler integration."""
    dts = jnp.diff(sim_input.time)
    # Check dts are all equal.
    dts = eqx.error_if(dts, jnp.any(jnp.abs(dts - dts[0]) > 1e-10), "Time steps must be uniform.")

    dt = dts[0]

    def _euler_step(carry, t):
        state = carry
        params_resolved = resolve_paths(sim_input.params, t)
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
            "state": state,
            "output": out,
            "params": params_resolved,
        }

        return state_next, output_data

    _, outputs = jax.lax.scan(_euler_step, sim_input.initial_state, xs=sim_input.time)

    return outputs
