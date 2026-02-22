import popsim.simulate as simulate
import pytest
import chex
import typing
from popsim import TimeDepModule, discrete_time_field, discrete_no_save_field
from popsim.modules.module_examples import DiscreteTimeExample, HybridExample, ExampleDisruptedState
import jax.numpy as jnp
from popsim.xarray_utils import time_and_pytree_to_xarray, solution_to_xarray, DEFAULT_SIM_DIM_NAME
import xarray as xr
from jaxtyping import Array
import jax
from popsim.input_utils import input_specs_to_paths
from popsim.interp import InterpType
from popsim.tree_util import tree_transpose
from popsim.sim_utils import make_time_base
from popsim.ml.utils import pad_time_with_epsilon

from popsim.simulate import SimInput, StepperType, _simple_euler_simulate, _simple_euler_simulate_uniform_timestep


class ContinuousTimeModule(TimeDepModule):
    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        x1: float
        x2: float

    @chex.dataclass
    class Output:
        y: float

    @chex.dataclass
    class Inputs:
        z: float = 0.0

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        state_dot = ContinuousTimeModule.State(x1=-state.x1, x2=-1.0 * state.x2)
        out = ContinuousTimeModule.Output(y=xr.apply_ufunc(jnp.abs, state.x1))
        return state_dot, out

@pytest.fixture
def pure_continuous_time_module():
    module = ContinuousTimeModule(config=ContinuousTimeModule.Config())
    initial_state = ContinuousTimeModule.State(x1=-1.0, x2=1.0)
    return module, initial_state

@pytest.fixture
def pure_discrete_time_module():
    module = DiscreteTimeExample(config=DiscreteTimeExample.Config(disruptivity_threshold=0.9))
    time_base = simulate.make_time_base(0.0, 1.0, 1e-3)
    initial_state = DiscreteTimeExample.State(disrupted_state=ExampleDisruptedState.NOT_DISRUPTED)
    inputs = DiscreteTimeExample.Inputs(disruptivity={0.0: 0.1, 0.5: 1.0, 0.75: 0.0})  # Time-dependent disruptivity.
    return module, time_base, initial_state, inputs

@pytest.fixture
def hybrid_time_module():
    module = HybridExample()
    time_base = simulate.make_time_base(0.0, 10.0, 1e-3)
    initial_state = HybridExample.State(y=0.0, sign=1)
    final_lim_mag = 0.01
    inputs = HybridExample.Inputs(speed=1.0, ylims=({0.0: -1.0, 10.0: -final_lim_mag}, {0.0: 1.0, 10.0: final_lim_mag}))
    return module, time_base, initial_state, inputs

@pytest.fixture
def xarray_partial_state():
    module = ContinuousTimeModule(config=ContinuousTimeModule.Config())
    initial_state = ContinuousTimeModule.State(x1=-1.0, x2=xr.DataArray(jnp.array([1.0, 2.0]), dims=["dim1"]))
    return module, initial_state

@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", [StepperType.SIMPLE_EULER, StepperType.DIFFRAX_TSIT5])
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_continuous_module(pure_continuous_time_module, return_xarray, stepper_type, multi_sim):
    module, initial_state = pure_continuous_time_module

    time_base = simulate.make_time_base(0.0, 10.0, 1e-3)

    if multi_sim:
        sim_inputs = [SimInput(time=time_base, initial_state=initial_state, inputs=ContinuousTimeModule.Inputs(z=0.0)), SimInput(time=time_base, initial_state=initial_state, inputs=ContinuousTimeModule.Inputs(z=1.0))]
    else:
        sim_inputs = SimInput(time=time_base, initial_state=initial_state, inputs=ContinuousTimeModule.Inputs(z=0.0))

    sol = simulate.simulate(module, sim_inputs, return_xarray=return_xarray, stepper_type=stepper_type)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type in (simulate.StepperType.SIMPLE_EULER, simulate.StepperType.SIMPLE_EULER_UNIFORM):
            sol = time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim)
        elif stepper_type in (simulate.StepperType.DIFFRAX_EULER, simulate.StepperType.DIFFRAX_TSIT5, simulate.StepperType.DIFFRAX_DOPRI5):
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
    
    
    # After 10 seconds, expect a significant amount of exponential decay of the state.
    assert (jnp.abs(sol["state.x1"].isel(time=-1).values) < 1.1 * jnp.exp(-jnp.max(time_base)) * jnp.abs(initial_state.x1)).all()

    # Expect that y is the absolute value of x.
    assert jnp.allclose(sol["output.y"].values, jnp.abs(sol["state.x1"].values))

    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert DEFAULT_SIM_DIM_NAME in sol.dims


@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", [StepperType.SIMPLE_EULER, StepperType.DIFFRAX_TSIT5])
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_discrete(pure_discrete_time_module, return_xarray, stepper_type, multi_sim):
    module, time_base, initial_state, inputs = pure_discrete_time_module

    sim_inputs = SimInput(time=time_base, initial_state=initial_state, inputs=inputs)
    if multi_sim:
        sim_inputs = [sim_inputs, sim_inputs]

    # Expect an error if the stepper type is Diffrax.
    if stepper_type in [simulate.StepperType.DIFFRAX_EULER, simulate.StepperType.DIFFRAX_TSIT5, simulate.StepperType.DIFFRAX_DOPRI5]:
        with pytest.raises(ValueError):
            sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)
        return
    else:
        sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type in (simulate.StepperType.SIMPLE_EULER, simulate.StepperType.SIMPLE_EULER_UNIFORM):
            sol = time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim)
        elif stepper_type in (simulate.StepperType.DIFFRAX_EULER, simulate.StepperType.DIFFRAX_TSIT5, simulate.StepperType.DIFFRAX_DOPRI5):
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
        
    # Check not disrupted at the start.
    assert (sol['state.disrupted_state'].sel(time=slice(0.0, 0.4)) == ExampleDisruptedState.NOT_DISRUPTED).all()

    # Check disrupted for the latter half.
    assert (sol['state.disrupted_state'].sel(time=slice(0.5, 1.0)) == ExampleDisruptedState.DISRUPTED).all()

@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", [StepperType.SIMPLE_EULER, StepperType.DIFFRAX_TSIT5])
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_hybrid_module(hybrid_time_module, return_xarray, stepper_type, multi_sim):
    module, time_base, initial_state, inputs = hybrid_time_module
    sim_inputs = SimInput(time=time_base, initial_state=initial_state, inputs=inputs)
    if multi_sim:
        sim_inputs = [sim_inputs, sim_inputs]

    # Expect an error if the stepper type is Diffrax.
    if stepper_type in [simulate.StepperType.DIFFRAX_EULER, simulate.StepperType.DIFFRAX_TSIT5, simulate.StepperType.DIFFRAX_DOPRI5]:
        with pytest.raises(ValueError):
            sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)
        return
    else:
        sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type in (simulate.StepperType.SIMPLE_EULER, simulate.StepperType.SIMPLE_EULER_UNIFORM):
            sol = time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim)
        elif stepper_type in (simulate.StepperType.DIFFRAX_EULER, simulate.StepperType.DIFFRAX_TSIT5, simulate.StepperType.DIFFRAX_DOPRI5):
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
    
    pad = 0.01
    assert (sol["state.y"] <= sol["inputs.ylims.1"] + pad).all()

    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert DEFAULT_SIM_DIM_NAME in sol.dims

@pytest.mark.parametrize("stepper_type", [StepperType.SIMPLE_EULER, StepperType.DIFFRAX_TSIT5])
@pytest.mark.parametrize("multi_sim", [True, False])
def test_xarray_partial_state(xarray_partial_state, stepper_type, multi_sim):
    module, initial_state = xarray_partial_state
    time_base = simulate.make_time_base(0.0, 10.0, 1e-3)
    if multi_sim:
        sim_inputs = [SimInput(time=time_base, initial_state=initial_state, inputs=ContinuousTimeModule.Inputs(z=0.0)), SimInput(time=time_base, initial_state=initial_state, inputs=ContinuousTimeModule.Inputs(z=1.0))]
    else:
        sim_inputs = SimInput(time=time_base, initial_state=initial_state, inputs=ContinuousTimeModule.Inputs(z=0.0))

    sol = simulate.simulate(module, sim_inputs, return_xarray=True, stepper_type=stepper_type)

    if multi_sim:
        assert sol["state.x1"].dims == (DEFAULT_SIM_DIM_NAME, "time")
        assert sol["state.x2"].dims == (DEFAULT_SIM_DIM_NAME, "time", "dim1")
    else:
        assert sol["state.x1"].dims == ("time", )
        assert sol["state.x2"].dims == ("time", "dim1")

def test_disable_record_state():
    """
    Test a module where recording the state would result OOM to see if disabling record_state works.
    """

    class MemoryHogExample(TimeDepModule):

        @chex.dataclass
        class State:
            big_array: Array = discrete_time_field()

        @chex.dataclass
        class Inputs:
            pass

        @chex.dataclass
        class Output:
            out: float

        def __call__(
            self, state: "MemoryHogExample.State", inputs: "MemoryHogExample.Inputs"
        ) -> tuple["MemoryHogExample.State", "MemoryHogExample.Output"]:
            state_out = MemoryHogExample.State(big_array=state.big_array)
            out = MemoryHogExample.Output(out=state.big_array[0])
            return state_out, out

    # 800MB array.
    array_size = int(1e8 if jax.config.jax_enable_x64 else 2e8)
    state = MemoryHogExample.State(big_array=jnp.ones(array_size))
    assert state.big_array.nbytes == 8e8

    # 800MB * 50 time steps = 80GB (should crash most computers).
    ts = jnp.linspace(0, 1, 100)

    module = MemoryHogExample()

    out = simulate.simulate(module, SimInput(time=ts, initial_state=state, inputs=MemoryHogExample.Inputs()), record_state=False)

def test_run_single_timestep(hybrid_time_module):
    module, time_base, initial_state, inputs = hybrid_time_module
    n_steps = time_base.size - 1
    dts = jnp.diff(time_base)
    state = initial_state
    inputs = HybridExample.Inputs(speed=1.0, ylims=(-1.0, 1.0))
    
    for i in range(n_steps):
        state, _ = simulate.single_step(module, state, inputs, dts[i])

    # Test that the final time step is the same as if we call simulate.simulate.
    out = simulate.simulate(module, SimInput(time=time_base, initial_state=initial_state, inputs=inputs), stepper_type=simulate.StepperType.SIMPLE_EULER, return_xarray=False)
    
    assert out['state'].y[-1] == state.y
    assert out['state'].sign[-1] == state.sign

def test_run_single_timestep_uniform(hybrid_time_module):
    module, time_base, initial_state, inputs = hybrid_time_module
    n_steps = time_base.size - 1
    dt = time_base[1] - time_base[0]
    state = initial_state
    inputs = HybridExample.Inputs(speed=1.0, ylims=(-1.0, 1.0))
    
    for _ in range(n_steps):
        state, _ = simulate.single_step(module, state, inputs, dt)

    # Test that the final time step is the same as if we call simulate.simulate.
    out = simulate.simulate(module, SimInput(time=time_base, initial_state=initial_state, inputs=inputs), stepper_type=simulate.StepperType.SIMPLE_EULER_UNIFORM, return_xarray=False)
    
    assert out['state'].y[-1] == state.y
    assert out['state'].sign[-1] == state.sign

def test_simple_euler_simulate_variable_timestep_match(pure_continuous_time_module):
    # Ensure the variable time step produces the same output as the original fixed time step implementation
    module, initial_state = pure_continuous_time_module
    dt = 0.01
    t_final = 10.0
    time_base = make_time_base(0.0, t_final, dt)

    sim_inputs = SimInput(
        time=time_base,
        initial_state=initial_state,
        inputs=ContinuousTimeModule.Inputs(z=0.0),
    )

    def _simulate(simulate_fun, sim_inputs):
        if not isinstance(sim_inputs, typing.Sequence):
            sim_inputs = [sim_inputs]
        sim_inputs = input_specs_to_paths(sim_inputs=sim_inputs, interp_type=InterpType.LINEAR)
        sim_inputs_vectorized = tree_transpose(sim_inputs, DEFAULT_SIM_DIM_NAME)
        sol = simulate_fun(module, sim_inputs_vectorized)
        return time_and_pytree_to_xarray(sim_inputs_vectorized.time, sol)
    
    sol_fixed = _simulate(_simple_euler_simulate_uniform_timestep, sim_inputs)
    sol_variable = _simulate(_simple_euler_simulate, sim_inputs)

    assert jnp.allclose(sol_fixed["state.x1"].values, sol_variable["state.x1"].values)
    assert jnp.allclose(sol_fixed["state.x2"].values, sol_variable["state.x2"].values)
    assert jnp.allclose(sol_fixed["output.y"].values, sol_variable["output.y"].values)

def test_simple_euler_simulate_epsilon_dt(pure_continuous_time_module):
    # Ensure epsilon timesteps at the end of a uniform simulation don't cause stuff to get too wonky
    module, initial_state = pure_continuous_time_module
    dt = 0.01
    t_final = 10.0
    normal_times = make_time_base(0.0, t_final, dt)
    nan_times = jnp.ones((10,)) * jnp.nan
    time_base = pad_time_with_epsilon(jnp.concatenate([normal_times, nan_times]))

    sim_inputs = SimInput(
        time=time_base,
        initial_state=initial_state,
        inputs=ContinuousTimeModule.Inputs(z=0.0),
    )

    def _simulate(simulate_fun, sim_inputs):
        if not isinstance(sim_inputs, typing.Sequence):
            sim_inputs = [sim_inputs]
        sim_inputs = input_specs_to_paths(sim_inputs=sim_inputs, interp_type=InterpType.LINEAR)
        sim_inputs_vectorized = tree_transpose(sim_inputs, DEFAULT_SIM_DIM_NAME)
        sol = simulate_fun(module, sim_inputs_vectorized)
        return time_and_pytree_to_xarray(sim_inputs_vectorized.time, sol)
    
    sol = _simulate(_simple_euler_simulate, sim_inputs)

    # Ensure no nans in output
    assert not jnp.isnan(sol["state.x1"].data).any()
    assert not jnp.isnan(sol["state.x2"].data).any()
    assert not jnp.isnan(sol["output.y"].data).any()

    # Ensure the values in the nan section are close-ish to the final value of the normal section
    last_normal_index = normal_times.size - 1
    ref_x1 = sol["state.x1"].isel(time=last_normal_index).values
    ref_x2 = sol["state.x2"].isel(time=last_normal_index).values
    
    sol_x1 = sol["state.x1"].isel(time=slice(last_normal_index + 1, None)).values
    sol_x2 = sol["state.x2"].isel(time=slice(last_normal_index + 1, None)).values

    assert jnp.allclose(sol_x1, ref_x1, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(sol_x2, ref_x2, rtol=1e-5, atol=1e-5)