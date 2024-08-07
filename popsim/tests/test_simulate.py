import popsim.simulate as simulate
import pytest
import chex
from popsim import ModuleBase, discrete_time_field
import jax.numpy as jnp
from popsim.xarray_utils import time_and_pytree_to_xarray, solution_to_xarray

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
def hybrid_time_module():
    @chex.dataclass
    class HybridTimeModule(ModuleBase):
        @chex.dataclass
        class Config:
            pass

        @chex.dataclass
        class State:
            x: int = discrete_time_field()
            y: float

        @chex.dataclass
        class Output:
            z: float

        @chex.dataclass
        class Params:
            a: float = 0

        config: Config

        def __init__(self, config):
            self.config = config

        def __call__(self, state: State, params: Params) -> tuple[State, Output]:
            state_dot = HybridTimeModule.State(x=state.x + 1, y=-state.y)
            out = HybridTimeModule.Output(z=jnp.abs(state.y))
            return state_dot, out
    
    initial_state = HybridTimeModule.State(x=0, y=-1.0)
    return HybridTimeModule, initial_state

@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_continuous_module(pure_continuous_time_module, return_xarray, stepper_type, multi_sim):
    ContinuousTimeModule, initial_state = pure_continuous_time_module

    module = ContinuousTimeModule(config=ContinuousTimeModule.Config())

    if multi_sim:
        params = [ContinuousTimeModule.Params(z=0.0), ContinuousTimeModule.Params(z=1.0)]
    else:
        params = ContinuousTimeModule.Params(z=0.0)

    ts = jnp.linspace(0.0, 10.0, 100)
    sol = simulate.simulate(module, ts, initial_state, params, stepper_type=stepper_type, return_xarray=return_xarray)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type == simulate.StepperType.SIMPLE_EULER:
            sol = time_and_pytree_to_xarray(ts, sol, multi_simulation=multi_sim)
        elif stepper_type == simulate.StepperType.DIFFRAX:
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
    
    
    # After 10 seconds, expect a significant amount of exponential decay of the state.
    assert (jnp.abs(sol["state.x"].isel(time=-1).values) < 1.1 * jnp.exp(-jnp.max(ts)) * jnp.abs(initial_state.x)).all()

    # Expect that y is the absolute value of x.
    assert jnp.allclose(sol["output.y"].values, jnp.abs(sol["state.x"].values))

    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert "simulation" in sol.dims

@pytest.mark.parametrize("return_xarray", [True, False])
@pytest.mark.parametrize("stepper_type", list(simulate.StepperType))
@pytest.mark.parametrize("multi_sim", [True, False])
def test_simulate_hybrid_module(hybrid_time_module, return_xarray, stepper_type, multi_sim):
    HybridTimeModule, initial_state = hybrid_time_module

    module = HybridTimeModule(config=HybridTimeModule.Config())

    if multi_sim:
        params = [HybridTimeModule.Params(a=0.0), HybridTimeModule.Params(a=1.0)]
    else:
        params = HybridTimeModule.Params()
    ts = jnp.linspace(0.0, 10.0, 100)

    # Expect an error if the stepper type is Diffrax.
    if stepper_type == simulate.StepperType.DIFFRAX:
        with pytest.raises(ValueError):
            sol = simulate.simulate(module, ts, initial_state, params, stepper_type=stepper_type, return_xarray=return_xarray)
        return
    else:
        sol = simulate.simulate(module, ts, initial_state, params, stepper_type=stepper_type, return_xarray=return_xarray)

    # In the cases where we don't return an xarray, convert the solution to an xarray for comparison.
    if return_xarray == False:
        if stepper_type == simulate.StepperType.SIMPLE_EULER:
            sol = time_and_pytree_to_xarray(ts, sol, multi_simulation=multi_sim)
        elif stepper_type == simulate.StepperType.DIFFRAX:
            sol = solution_to_xarray(sol, multi_simulation=multi_sim)
        else:
            raise ValueError("Stepper type not recognized.")
    
    # After 10 seconds, expect a significant amount of exponential decay of the state.
    assert (jnp.abs(sol["state.y"].isel(time=-1).values) < 1.1 * jnp.exp(-jnp.max(ts)) * jnp.abs(initial_state.y)).all()

    # Expect that z is the absolute value of y.
    assert jnp.allclose(sol["output.z"].values, jnp.abs(sol["state.y"].values))

    # Expect x to be incremented by 1 at each time step.
    assert jnp.allclose(sol["state.x"].values, jnp.arange(0, len(ts)))


    # If multi_sim, check that there is a simulation dimension.
    if multi_sim:
        assert "simulation" in sol.dims