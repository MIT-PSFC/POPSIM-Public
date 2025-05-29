import popsim.simulate as simulate
import pytest
import chex
from popsim import TimeDepModule, discrete_time_field, discrete_no_save_field
from popsim.modules.module_examples import DiscreteTimeExample, HybridExample, ExampleDisruptedState
import jax.numpy as jnp
from popsim.xarray_utils import time_and_pytree_to_xarray, solution_to_xarray, DEFAULT_SIM_DIM_NAME
from popsim.simulate import SimInput
import xarray as xr
from jaxtyping import Array
from jaxlib.xla_extension import XlaRuntimeError
import equinox as eqx

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
@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
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
        if stepper_type == simulate.StepperType.SIMPLE_EULER:
            sol = time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim)
        elif stepper_type == simulate.StepperType.DIFFRAX:
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
@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_discrete(pure_discrete_time_module, return_xarray, stepper_type, multi_sim):
    module, time_base, initial_state, inputs = pure_discrete_time_module

    sim_inputs = SimInput(time=time_base, initial_state=initial_state, inputs=inputs)
    if multi_sim:
        sim_inputs = [sim_inputs, sim_inputs]

    # Expect an error if the stepper type is Diffrax.
    if stepper_type == simulate.StepperType.DIFFRAX:
        with pytest.raises(ValueError):
            sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)
        return
    else:
        sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type == simulate.StepperType.SIMPLE_EULER:
            sol = time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim)
        elif stepper_type == simulate.StepperType.DIFFRAX:
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
        
    # Check not disrupted at the start.
    assert (sol['state.disrupted_state'].sel(time=slice(0.0, 0.4)) == ExampleDisruptedState.NOT_DISRUPTED).all()

    # Check disrupted for the latter half.
    assert (sol['state.disrupted_state'].sel(time=slice(0.5, 1.0)) == ExampleDisruptedState.DISRUPTED).all()

@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_hybrid_module(hybrid_time_module, return_xarray, stepper_type, multi_sim):
    module, time_base, initial_state, inputs = hybrid_time_module
    sim_inputs = SimInput(time=time_base, initial_state=initial_state, inputs=inputs)
    if multi_sim:
        sim_inputs = [sim_inputs, sim_inputs]

    # Expect an error if the stepper type is Diffrax.
    if stepper_type == simulate.StepperType.DIFFRAX:
        with pytest.raises(ValueError):
            sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)
        return
    else:
        sol = simulate.simulate(module, sim_inputs, stepper_type=stepper_type, return_xarray=return_xarray)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type == simulate.StepperType.SIMPLE_EULER:
            sol = time_and_pytree_to_xarray(time_base, sol, multi_simulation=multi_sim)
        elif stepper_type == simulate.StepperType.DIFFRAX:
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
    
    pad = 0.01
    assert (sol["state.y"] <= sol["inputs.ylims.1"] + pad).all()

    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert DEFAULT_SIM_DIM_NAME in sol.dims

@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
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
    state = MemoryHogExample.State(big_array=jnp.ones(int(1e8)))
    assert state.big_array.nbytes == 8e8

    # 800MB * 50 time steps = 80GB (should crash most computers).
    ts = jnp.linspace(0, 1, 100)

    module = MemoryHogExample()

    out = simulate.simulate(module, SimInput(time=ts, initial_state=state, inputs=MemoryHogExample.Inputs()), record_state=False)

def test_run_single_timestep(hybrid_time_module):
    module, time_base, initial_state, inputs = hybrid_time_module
    n_steps = time_base.size - 1
    dt = time_base[1] - time_base[0]
    state = initial_state
    inputs = HybridExample.Inputs(speed=1.0, ylims=(-1.0, 1.0))
    
    for _ in range(n_steps):
        state, _ = simulate.simple_step(module, state, inputs, dt)

    # Test that the final time step is the same as if we call simulate.simulate.
    out = simulate.simulate(module, SimInput(time=time_base, initial_state=initial_state, inputs=inputs), stepper_type=simulate.StepperType.SIMPLE_EULER, return_xarray=False)
    
    assert out['state'].y[-1] == state.y
    assert out['state'].sign[-1] == state.sign
