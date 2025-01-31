from enum import IntEnum

import chex
import jax.numpy as jnp

from popsim import ModuleBase, discrete_time_field
from popsim.logic_utils import select_w_tuples

"""
Contains three example modules for automated testing and educational purposes.
See docs/notebooks/intro_to_modules.ipynb for a tutorial on how to use these modules.
"""


@chex.dataclass
class BasicLorenz(ModuleBase):
    @chex.dataclass
    class Config:
        # Define the configuration variables for the module.
        xlims: tuple[float, float] = (-20.0, 20.0)  # Lower and upper limits for x
        ylims: tuple[float, float] = (-20.0, 20.0)  # Lower and upper limits for y
        zlims: tuple[float, float] = (-20.0, 20.0)  # Lower and upper limits for z

    @chex.dataclass
    class State:
        # Define the state variables for the module.
        x: float
        y: float
        z: float

    @chex.dataclass
    class Output:
        # Define the output variables that will be returned by the module.
        distance_from_origin: float  # L2 norm of the state

    @chex.dataclass
    class Inputs:
        # Define the (possibly time-dependent) inputs for the module.
        rho: float
        sigma: float
        beta: float

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        def val_in_limits(val, lims):
            # Function to check a single variable is within the limits
            return jnp.logical_and(val > lims[0], val < lims[1])

        # Check if all variables are within the limits.
        in_limits = jnp.all(
            jnp.array(
                [
                    val_in_limits(state.x, self.config.xlims),
                    val_in_limits(state.y, self.config.ylims),
                    val_in_limits(state.z, self.config.zlims),
                ]
            )
        )

        # Lorenz equations.
        x_dot = inputs.rho * (state.y - state.x)
        y_dot = state.x * (inputs.sigma - state.z) - state.y
        z_dot = state.x * state.y - inputs.beta * state.z

        # If the state is out of limits, set the derivatives to zero.
        state_dot = BasicLorenz.State(
            x=jnp.where(in_limits, x_dot, 0.0),
            y=jnp.where(in_limits, y_dot, 0.0),
            z=jnp.where(in_limits, z_dot, 0.0),
        )

        # Make an output.
        out = BasicLorenz.Output(distance_from_origin=jnp.sqrt(state.x**2 + state.y**2 + state.z**2))
        return state_dot, out


class ExampleDisruptedState(IntEnum):
    NOT_DISRUPTED = 0
    DISRUPTED = 1


@chex.dataclass
class DiscreteTimeExample(ModuleBase):
    @chex.dataclass
    class Config:
        disruptivity_threshold: float

    @chex.dataclass
    class State:
        disrupted_state: ExampleDisruptedState = discrete_time_field(default=ExampleDisruptedState.NOT_DISRUPTED)

    @chex.dataclass
    class Inputs:
        disruptivity: float

    @chex.dataclass
    class Output:
        pass

    config: Config

    def __init__(self, config):
        self.config = config

    def __call__(
        self, state: "DiscreteTimeExample.State", inputs: "DiscreteTimeExample.Inputs"
    ) -> tuple["DiscreteTimeExample.State", "DiscreteTimeExample.Output"]:
        # If the disruptivity is above the threshold, set the state to DISRUPTED.
        # If the state is already DISRUPTED, keep it that way, regardless of the disruptivity.
        disruptivity_above_threshold = inputs.disruptivity > self.config.disruptivity_threshold
        next_state = jnp.where(
            jnp.logical_or(disruptivity_above_threshold, state.disrupted_state == ExampleDisruptedState.DISRUPTED),
            ExampleDisruptedState.DISRUPTED,
            ExampleDisruptedState.NOT_DISRUPTED,
        )

        # NOTE: this is not state_dot anymore, but the state at the next time step.
        state_out = DiscreteTimeExample.State(disrupted_state=next_state)
        out = DiscreteTimeExample.Output()
        return state_out, out


@chex.dataclass
class HybridExample(ModuleBase):
    @chex.dataclass
    class Config:
        pass

    @chex.dataclass
    class State:
        y: float  # Continuous-time state variable for the position.
        sign: int = discrete_time_field()  # Discrete-time state variable for the sign of the time derivative.

    @chex.dataclass
    class Inputs:
        speed: float
        ylims: tuple[float, float]

    @chex.dataclass
    class Output:
        pass

    def __call__(
        self, state: "HybridExample.State", inputs: "HybridExample.Inputs"
    ) -> tuple["HybridExample.State", "HybridExample.Output"]:
        # If the state is at the upper limit, set the sign to -1.
        # If the state is at the lower limit, set the sign to 1.
        # Otherwise, keep the sign the same.
        at_upper_lim = state.y >= inputs.ylims[1]
        at_lower_lim = state.y <= inputs.ylims[0]

        conditions_and_choices = [
            (at_upper_lim, -1),
            (at_lower_lim, 1),
            (jnp.logical_and(~at_upper_lim, ~at_lower_lim), state.sign),
        ]
        next_sign = select_w_tuples(conditions_and_choices, default=1)

        # NOTE: the output is no longer purely a "state_dot" or the state at the next time step, but rather a mix.
        # The continuous part of the state is still a time derivative, but the discrete part is the state value at
        # the next time step of the simulation.
        ydot = inputs.speed * state.sign
        state_out = HybridExample.State(
            y=ydot,
            sign=next_sign,
        )
        out = HybridExample.Output()
        return state_out, out
