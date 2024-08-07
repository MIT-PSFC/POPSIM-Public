from popsim.modules.pid import PIDController
import jax.numpy as jnp
from popsim.simulate import simulate
from popsim.interp import interp

def test_pid_controller():
    # Initialize the PID controller
    config = PIDController.Config(Kp=1.0, Ki=0.1, Kd=0.05, dt=0.1, output_min=-10, output_max=10)
    pid = PIDController(config=config)

    # Create initial state with empty error history
    initial_state = PIDController.State(integrated_error=0.0, previous_error=0.0)
    params = PIDController.Params(setpoint=5.0, measurement=0.0)

    # Run the PID controller once.
    state_out, output = pid(initial_state, params)

    assert isinstance(state_out, PIDController.State)
    assert isinstance(output, PIDController.Output)

    # Expect the control output to be the proportional term only for this first iteration.
    expected_control = config.Kp * 5.0
    assert jnp.isclose(output.control, expected_control, atol=1e-6)

    # Test output limiting
    params_max = PIDController.Params(setpoint=100.0, measurement=0.0)
    _, output_max = pid(initial_state, params_max)
    assert output_max.control == config.output_max

    params_min = PIDController.Params(setpoint=-100.0, measurement=0.0)
    _, output_min = pid(initial_state, params_min)
    assert output_min.control == config.output_min

    # Test that we can run "simulate" with time varying setpoint and measurement.
    ts = 0.1 * jnp.arange(100)
    params = PIDController.Params(setpoint=interp(ts, jnp.sin(ts)), measurement=interp(ts, jnp.cos(ts)))
    ds = simulate(pid, ts, initial_state, params, return_xarray=True)
