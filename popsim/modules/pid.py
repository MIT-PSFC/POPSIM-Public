import chex
import jax.numpy as jnp
from jaxtyping import Array

from popsim import ModuleBase, discrete_time_field


@chex.dataclass
class PIDController(ModuleBase):
    @chex.dataclass
    class Config:
        Kp: float
        Ki: float
        Kd: float
        dt: float
        output_min: float = -float("inf")
        output_max: float = float("inf")

    @chex.dataclass
    class State:
        error_history: Array = discrete_time_field()

    @chex.dataclass
    class Output:
        control: float
        aux_data: dict

    @chex.dataclass
    class Params:
        setpoint: float
        measurement: float

    config: Config

    def __init__(self, config=None):
        self.config = config or self.Config()

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        error = params.setpoint - params.measurement

        # Proportional term
        P = self.config.Kp * error

        # Integral term (using finite time horizon)
        I = self.config.Ki * jnp.sum(state.error_history) * self.config.dt  # noqa: E741

        # Derivative term. If the previous error is zero, the derivative term is zero to handle the initialization case.
        previous_error = state.error_history[0]
        D = jnp.where(previous_error == 0.0, 0.0, self.config.Kd * (error - previous_error) / self.config.dt)

        # Calculate control output
        control = P + I + D

        # Apply output limits
        control = jnp.clip(control, self.config.output_min, self.config.output_max)

        # Update state
        new_error_history = jnp.concatenate([jnp.array([error]), state.error_history[:-1]])
        next_state = PIDController.State(error_history=new_error_history, previous_error=error)

        output = PIDController.Output(control=control, aux_data={"P": P, "I": I, "D": D})

        return next_state, output
