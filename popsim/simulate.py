import typing
import warnings
from enum import IntEnum
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import PyTree
from loguru import logger

from popsim import TimeDepModule, config
from popsim.array_utils import min_greater_than_thresh
from popsim.field_labels import partition_discrete_cont, partition_save_no_save
from popsim.input_utils import input_specs_to_paths
from popsim.interp import InterpType, resolve_paths
from popsim.modules.prng import PRNGModule
from popsim.sim_utils import (
    CombinatorialCases,  # . Import is used to allow the user to import this function from this module.
    MultiCases,  # . Import is used to allow the user to import this function from this module.
    SimInput,  # noqa: F401. Import is used to allow the user to import this function from this module.
    make_time_base,  # noqa: F401
)
from popsim.tree_util import get_instances_from_tree_leaves, tree_transpose
from popsim.utils import time_epsilon
from popsim.xarray_utils import (
    DEFAULT_SIM_DIM_NAME,
    DEFAULT_TIME_DIM_NAME,
    add_dim_to_vars,
    run_function_with_dim_removed,
    solution_to_xarray,
    time_and_pytree_to_xarray,
)


class StepperType(IntEnum):
    SIMPLE_EULER = 0
    DIFFRAX_EULER = 1
    DIFFRAX_TSIT5 = 2
    DIFFRAX_DOPRI5 = 3


def _check_sim_inputs(module: TimeDepModule, sim_inputs: typing.Sequence[SimInput], stepper_type: StepperType):
    """Perform checks of the simulation inputs."""

    # Check if the state has any discrete components when using Diffrax. If so, raise an error.
    if stepper_type in [StepperType.DIFFRAX_EULER, StepperType.DIFFRAX_TSIT5, StepperType.DIFFRAX_DOPRI5]:

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


def generate_save_output(state: PyTree, inputs: PyTree, output: PyTree, record_state: bool = True) -> PyTree:
    """Generate the output data for a simulation."""
    out = {"output": output, "inputs": inputs}
    if record_state:
        out["state"] = state

    for key, val in out.items():
        save, _ = partition_save_no_save(val)
        out[key] = save
    return out


def simulate(
    module: TimeDepModule,
    sim_inputs: SimInput | typing.Sequence[SimInput],
    interp_type: InterpType = InterpType.LINEAR,
    return_xarray: bool = True,
    stepper_type: StepperType = StepperType.SIMPLE_EULER,
    record_state: bool = True,
) -> diffrax.Solution | xr.Dataset:
    """Public facing API for simulating a module.

    Args:
        module (TimeDepModule): the dynamics module to simulate.
        sim_inputs (typing.Union[SimInput, typing.Sequence[SimInput]]): simulation input.
        interp_type (InterpType, optional): interpolation method for inputs over time.
        return_xarray (bool, optional): whether to return a xr.Dataset or a diffrax.Solution. Defaults to True.
        stepper_type (StepperType, optional): stepper type to use. Defaults to StepperType.SIMPLE_EULER.
        record_state (bool, optional): whether to record the state at each time step. Defaults to True.

    Returns:
        typing.Union[diffrax.Solution, xr.Dataset]: simulation results.
    """
    if not isinstance(sim_inputs, typing.Sequence):
        sim_inputs = [sim_inputs]

    _check_sim_inputs(module, sim_inputs, stepper_type)

    # Resolve the input specifications to paths.
    sim_inputs = input_specs_to_paths(sim_inputs=sim_inputs, interp_type=interp_type)

    # Choose the simulation function based on the stepper type.
    if stepper_type == StepperType.SIMPLE_EULER:

        def simulate_fun(mod, inp):
            return _simple_euler_simulate(mod, inp, record_state=record_state)
    elif stepper_type == StepperType.DIFFRAX_EULER:

        def simulate_fun(mod, inp):
            return _diffrax_simulate(mod, inp, record_state=record_state, solver=diffrax.Euler())
    elif stepper_type == StepperType.DIFFRAX_TSIT5:

        def simulate_fun(mod, inp):
            return _diffrax_simulate(mod, inp, record_state=record_state, solver=diffrax.Tsit5())
    elif stepper_type == StepperType.DIFFRAX_DOPRI5:

        def simulate_fun(mod, inp):
            return _diffrax_simulate(mod, inp, record_state=record_state, solver=diffrax.Dopri5())
    else:
        raise ValueError("Stepper type not recognized.")

    # Vectorize the simulation inputs.
    sim_inputs_vectorized = tree_transpose(sim_inputs, DEFAULT_SIM_DIM_NAME)

    multi_sim = len(sim_inputs) > 1
    logger.info(f"Running {len(sim_inputs)} simulations.")

    # Perform the simulation.
    if multi_sim:
        sol = _vec_simulate(module, sim_inputs_vectorized, simulate_fun=simulate_fun)
    else:
        sol = simulate_fun(module, sim_inputs_vectorized)

    if stepper_type == StepperType.SIMPLE_EULER:
        return time_and_pytree_to_xarray(sim_inputs_vectorized.time, sol, multi_simulation=multi_sim) if return_xarray else sol
    elif stepper_type in [StepperType.DIFFRAX_EULER, StepperType.DIFFRAX_TSIT5, StepperType.DIFFRAX_DOPRI5]:
        return solution_to_xarray(sol, multi_simulation=multi_sim) if return_xarray else sol
    else:
        raise ValueError("Stepper type not recognized.")


@partial(jax.jit, static_argnames=("record_state"))
def single_step(module: TimeDepModule, state: PyTree, inputs: PyTree, dt: float, record_state: bool = True) -> tuple[PyTree, PyTree]:
    """Perform a single time step of a module. Continuous states are updated using Euler integration.

    Args:
        module (TimeDepModule): the module to simulate for a time step.
        state (PyTree): state structure of the module.
        inputs (PyTree): inputs structure of the module.
        dt (float): time step size for the Euler integration.
        record_state (bool, optional): Whether or not to record state in the output structure. Defaults to True.

    Returns:
        tuple[PyTree, PyTree]: (next_state, output_data)
    """
    # Make sure state + inputs have JAX arrays.
    state = jax.tree.map(lambda x: jnp.asarray(x), state)
    inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
    # Perform a single step.
    state_next, output_data = _single_step(module, state, inputs, dt, record_state=record_state)
    return state_next, output_data


def _single_step(module: TimeDepModule, state: PyTree, inputs: PyTree, dt: float, record_state: bool = True) -> tuple[PyTree, PyTree]:
    state_out, out = module(state, inputs)

    # Partition the state output tree into continuous (float, complex, and arrays of float + complex) and discrete parts (everything else).
    # The continuous parts are assumed to be state_dot. The discrete parts are assumed to be the next state.
    discrete_state_next, state_dot = partition_discrete_cont(state_out)

    def step_fn(x, xdot):
        return x + xdot * dt if xdot is not None else None

    # Perform an Euler step on the continuous part
    continuous_state_next = jax.tree.map(step_fn, state, state_dot)

    # Combine the next continuous state with the next discrete state
    state_next = eqx.combine(discrete_state_next, continuous_state_next)

    output_data = generate_save_output(state, inputs, out, record_state=record_state)

    return state_next, output_data


@eqx.filter_jit
def _vec_simulate(module: TimeDepModule, sim_input: SimInput, simulate_fun):
    """Perform a vectorized simulation."""
    sim_input_axes = jax.tree.map(lambda x: 0, sim_input)

    vec_sim_fun = jax.vmap(simulate_fun, in_axes=(None, sim_input_axes))

    sol = run_function_with_dim_removed(vec_sim_fun, (module, sim_input), DEFAULT_SIM_DIM_NAME)

    # Remove extraneous dimensions.
    sol = jax.tree.map(lambda x: jnp.squeeze(x), sol)
    return sol


@eqx.filter_jit
def _diffrax_simulate(
    module: TimeDepModule,
    sim_input: SimInput,
    record_state: bool = True,
    max_steps: int = config["DIFFRAX_MAX_STEPS"],
    solver: diffrax.AbstractAdaptiveSolver | None = None,
) -> diffrax.Solution:
    """Function for simulating a single case using diffrax. Default solver is Dopri5."""

    if solver is None:
        solver = diffrax.Tsit5()

    def module_f(t, y, inputs, return_aux=False):
        inputs_resolved = resolve_paths(inputs, t)
        state_dot, out = module(y, inputs_resolved)
        if return_aux:
            return out, inputs_resolved
        else:
            return state_dot

    # Function to save auxiliary information.
    def saveat_fn(t, y, args):
        output, inputs_resolved = module_f(t, y, args, return_aux=True)
        out = generate_save_output(y, inputs_resolved, output, record_state=record_state)
        return out

    # Get the minimum time step that is greater than the maximum time padding amount.
    dt0 = min_greater_than_thresh(jnp.diff(sim_input.time), jnp.max(time_epsilon(sim_input.time)))

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(module_f),
        solver=solver,
        t0=sim_input.time[0],
        t1=sim_input.time[-1],
        dt0=dt0,
        y0=sim_input.initial_state,
        args=sim_input.inputs,
        saveat=diffrax.SaveAt(ts=sim_input.time, fn=saveat_fn),
        max_steps=max_steps,
    )
    # Add simulation dimension to any xr.Variable instances.
    sol = add_dim_to_vars(sol, DEFAULT_TIME_DIM_NAME)
    return sol


@eqx.filter_jit
def _simple_euler_simulate_fixed_timestep(module: TimeDepModule, sim_input: SimInput, record_state: bool = True) -> PyTree:
    """Function for simulating a single case using simple Euler integration with fixed time steps."""
    dts = jnp.diff(sim_input.time)
    # Check dts are all equal.
    if jax.config.jax_enable_x64:
        dt_epsilon = 1e-10
    else:
        dt_epsilon = 1e-6

    dts = eqx.error_if(dts, jnp.any(jnp.abs(dts - dts[0]) > dt_epsilon), "Time steps must be uniform.")

    dt = dts[0]

    def _step(carry, t):
        """Perform a single Euler step."""
        state = carry
        inputs_resolved = resolve_paths(sim_input.inputs, t)
        state_next, outputs = _single_step(module, state, inputs_resolved, dt, record_state=record_state)
        return state_next, outputs

    if dts.size == 1:
        # We only have one time step, which makes things simpler.
        outputs = _step(sim_input.initial_state, sim_input.time)
    else:
        _, outputs = jax.lax.scan(_step, sim_input.initial_state, xs=sim_input.time)
    # Add simulation dimension to any xr.Variable instances.
    outputs = add_dim_to_vars(outputs, DEFAULT_TIME_DIM_NAME)
    return outputs


@eqx.filter_jit
def _simple_euler_simulate(module: TimeDepModule, sim_input: SimInput, record_state: bool = True) -> PyTree:
    """Function for simulating a single case using Euler integration with non-uniform time steps."""
    dts = jnp.diff(sim_input.time)

    def _step(carry, timing_info):
        """Perform a single Euler step."""
        state = carry
        inputs_resolved = resolve_paths(sim_input.inputs, timing_info["ts"])
        state_next, outputs = _single_step(module, state, inputs_resolved, timing_info["dts"], record_state=record_state)
        return state_next, outputs

    if dts.size == 1:
        # We only have one time step, which makes things simpler.
        outputs = _step(sim_input.initial_state, sim_input.time, dts[0])
    else:
        dt0 = min_greater_than_thresh(dts, jnp.max(time_epsilon(sim_input.time)))
        dts_padded = jnp.concatenate([jnp.array([dt0]), dts], axis=0)  # Pad dts to match time size.
        timing_info = {"ts": sim_input.time, "dts": dts_padded}
        _, outputs = jax.lax.scan(_step, sim_input.initial_state, xs=timing_info)
    # Add simulation dimension to any xr.Variable instances.
    outputs = add_dim_to_vars(outputs, DEFAULT_TIME_DIM_NAME)
    return outputs
