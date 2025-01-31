import chex
import jax.numpy as jnp

from popsim import ModuleBase, discrete_time_field


@chex.dataclass
class PIDController(ModuleBase):
    """A Basic PID Controller."""

    @chex.dataclass
    class Config:
        Kp: float  # Propotional gain [-]
        Ki: float  # Integral gain [-]
        Kd: float  # Derivative gain [-]
        dt: float  # Time step between control updates [s]
        output_min: float = -float("inf")  # Minimum control output
        output_max: float = float("inf")  # Maximum control output

    @chex.dataclass
    class State:
        integrated_error: float = 0.0  # Integral of the error over time
        previous_error: float = discrete_time_field(default=0.0)  # Previous error value
        uninitialized: bool = discrete_time_field(default=True)  # Flag to indicate if the controller has been initialized

    @chex.dataclass
    class Output:
        control: float  # Control output
        aux_data: dict  # Auxiliary data for debugging and analysis.

    @chex.dataclass
    class Inputs:
        setpoint: float  # Control setpoint to track
        measurement: float  # Measured value to apply feedback with
        feed_forward: float = 0.0  # Feed forward control signal that gets added to the PID output to deterimne Output.control.

    config: Config

    def __init__(self, config=None):
        self.config = config or self.Config()

    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        error = inputs.setpoint - inputs.measurement

        # Proportional term
        P = self.config.Kp * error

        # Integral term (using finite time horizon)
        I = self.config.Ki * state.integrated_error  # noqa: E741

        # Derivative term. If the previous error is zero, the derivative term is zero to handle the initialization case.
        D = jnp.where(state.uninitialized, 0.0, self.config.Kd * (error - state.previous_error) / self.config.dt)

        # Calculate control output
        pid_act = P + I + D

        # Apply output limits
        control = jnp.clip(pid_act + inputs.feed_forward, self.config.output_min, self.config.output_max)

        # The time derivative of integrated error is just error.
        integrated_error_dot = error
        state_out = PIDController.State(integrated_error=integrated_error_dot, previous_error=error, uninitialized=False)

        output = PIDController.Output(control=control, aux_data={"P": P, "I": I, "D": D, "PID_act": pid_act})

        return state_out, output
