import popsim.simulate as simulate
import pytest
import chex
from popsim import ModuleBase, discrete_time_field
from popsim.modules.module_examples import DiscreteTimeExample, HybridExample, ExampleDisruptedState
import jax.numpy as jnp
import jax
from popsim.xarray_utils import time_and_pytree_to_xarray, solution_to_xarray
from popsim.types import SimInput

@pytest.fixture
def pure_continuous_time_module():
    @chex.dataclass
    class ContinuousTimeModule(ModuleBase):
        @chex.dataclass
        class Config:
            pass

        @chex.dataclass
        class State:
            x: float

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
            state_dot = ContinuousTimeModule.State(x=-state.x)
            out = ContinuousTimeModule.Output(y=jnp.abs(state.x))
            return state_dot, out
        
    initial_state = ContinuousTimeModule.State(x=-1.0)
    
    return ContinuousTimeModule, initial_state

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

@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_continuous_module(pure_continuous_time_module, return_xarray, stepper_type, multi_sim):
    ContinuousTimeModule, initial_state = pure_continuous_time_module

    module = ContinuousTimeModule(config=ContinuousTimeModule.Config())

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
    assert (jnp.abs(sol["state.x"].isel(time=-1).values) < 1.1 * jnp.exp(-jnp.max(time_base)) * jnp.abs(initial_state.x)).all()

    # Expect that y is the absolute value of x.
    assert jnp.allclose(sol["output.y"].values, jnp.abs(sol["state.x"].values))

    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert "simulation" in sol.dims


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
        assert "simulation" in sol.dims