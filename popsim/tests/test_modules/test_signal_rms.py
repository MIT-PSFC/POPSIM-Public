import jax.numpy as jnp
from scipy.signal import sawtooth

from popsim.modules.signal_rms import SignalRMS


def test_signal_rms_default_state():

    default_state = SignalRMS.State()

def test_signal_rms_simple():
    signal_rms = SignalRMS()

    window_size = 4
    signal_rms_initial_state = SignalRMS.State(window=jnp.zeros(window_size))

    # Ensure the output remains 0 after giving the module 0 input.
    signal_rms_inputs = SignalRMS.Inputs(signal=0)
    state_out, output = signal_rms(signal_rms_initial_state, signal_rms_inputs)
    assert output.rms == 0

    # Give the module a constant signal of 1 for several time steps and ensure the output is continuously increasing
    for i in range(window_size):
        previous_output = output.rms
        signal_rms_inputs = SignalRMS.Inputs(signal=1)
        state_out, output = signal_rms(state_out, signal_rms_inputs)
        assert output.rms >= previous_output

    # Ensure final output is 1
    assert output.rms == 1


def test_signal_rms_known():
    # Give the module many time steps of a signal with known RMS and ensure the output is close to correct.
    
    window_size = 100
    signal_rms = SignalRMS()

    state_out = SignalRMS.State(window=jnp.zeros(window_size))
    x_vals = jnp.linspace(0, 10 * jnp.pi, 1000)

    # sin wave
    signal = jnp.sin(x_vals)
    for i, _ in enumerate(x_vals):
        signal_rms_inputs = SignalRMS.Inputs(signal=signal[i])
        state_out, output = signal_rms(state_out, signal_rms_inputs)
    assert jnp.isclose(output.rms, 1 / jnp.sqrt(2), atol=1e-3)

    # sawtooth wave
    signal = sawtooth(x_vals, width=0.5)
    for i, _ in enumerate(x_vals):
        signal_rms_inputs = SignalRMS.Inputs(signal=signal[i])
        state_out, output = signal_rms(state_out, signal_rms_inputs)
    assert jnp.isclose(output.rms, 1 / jnp.sqrt(3), atol=1e-3)