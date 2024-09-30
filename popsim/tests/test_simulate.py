import popsim.simulate as simulate
import pytest
import chex
from popsim import ModuleBase, discrete_time_field
from popsim.modules.module_examples import DiscreteTimeExample, HybridExample, ExampleDisruptedState
import jax.numpy as jnp
from popsim.xarray_utils import time_and_pytree_to_xarray, solution_to_xarray, DEFAULT_SIM_DIM_NAME
from popsim.simulate import SimInput
import xarray as xr

@chex.dataclass
class ContinuousTimeModule(ModuleBase):
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
    class Params:
        z: float = 0.0

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
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
    params = DiscreteTimeExample.Params(disruptivity={0.0: 0.1, 0.5: 1.0, 0.75: 0.0})  # Time-dependent disruptivity.
    return module, time_base, initial_state, params

@pytest.fixture
def hybrid_time_module():
    module = HybridExample()
    time_base = simulate.make_time_base(0.0, 10.0, 1e-3)
    initial_state = HybridExample.State(y=0.0, sign=1)
    final_lim_mag = 0.01
    params = HybridExample.Params(speed=1.0, ylims=({0.0: -1.0, 10.0: -final_lim_mag}, {0.0: 1.0, 10.0: final_lim_mag}))
    return module, time_base, initial_state, params

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
        sim_inputs = [SimInput(time=time_base, initial_state=initial_state, params=ContinuousTimeModule.Params(z=0.0)), SimInput(time=time_base, initial_state=initial_state, params=ContinuousTimeModule.Params(z=1.0))]
    else:
        sim_inputs = SimInput(time=time_base, initial_state=initial_state, params=ContinuousTimeModule.Params(z=0.0))

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
    module, time_base, initial_state, params = pure_discrete_time_module

    sim_inputs = SimInput(time=time_base, initial_state=initial_state, params=params)
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
    module, time_base, initial_state, params = hybrid_time_module
    sim_inputs = SimInput(time=time_base, initial_state=initial_state, params=params)
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
    assert (sol["state.y"] <= sol["params.ylims.1"] + pad).all()

    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert DEFAULT_SIM_DIM_NAME in sol.dims

@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
@pytest.mark.parametrize("multi_sim", [True, False])
def test_xarray_partial_state(xarray_partial_state, stepper_type, multi_sim):
    module, initial_state = xarray_partial_state
    time_base = simulate.make_time_base(0.0, 10.0, 1e-3)
    if multi_sim:
        sim_inputs = [SimInput(time=time_base, initial_state=initial_state, params=ContinuousTimeModule.Params(z=0.0)), SimInput(time=time_base, initial_state=initial_state, params=ContinuousTimeModule.Params(z=1.0))]
    else:
        sim_inputs = SimInput(time=time_base, initial_state=initial_state, params=ContinuousTimeModule.Params(z=0.0))

    sol = simulate.simulate(module, sim_inputs, return_xarray=True, stepper_type=stepper_type)

    if multi_sim:
        assert sol["state.x1"].dims == (DEFAULT_SIM_DIM_NAME, "time")
        assert sol["state.x2"].dims == (DEFAULT_SIM_DIM_NAME, "time", "dim1")
    else:
        assert sol["state.x1"].dims == ("time", )
        assert sol["state.x2"].dims == ("time", "dim1")