import jax.numpy as jnp

from popsim.modules.rtnewspec_mirror import RTNewSpecMirror

def test_calculated_rms_simple():
    """Test the RMS calculation of a simple sinusoidal signal with known phase offset."""

    probe_offset = jnp.pi/5

    config = RTNewSpecMirror.Config(
        d_theta=jnp.rad2deg(probe_offset),
        probe1_id="probe1",
        probe2_id="probe2",
        nsamples=2048,
        alpha=1.0,
        f_probe=60e3,
        nsmth=3,
        max_modes=2,
    )

    rtnewspec_mirror = RTNewSpecMirror(config=config)

    # Make repeating sinusoidal signals with 10 kHz frequency as if they were being sampled at 60 kHz
    time = jnp.arange(2048)/60e3
    probe1_signal = jnp.sin(2*jnp.pi*10e3*time)
    probe2_signal = jnp.sin(2*jnp.pi*10e3*time - probe_offset)

    # Calculate the RMS values directly
    rms_values = rtnewspec_mirror.calculate_rms(probe1_signal, probe2_signal)

    # Check that the output is large for n=1 but small for n=2
    assert rms_values[1] > 0.9
    assert rms_values[2] < 0.1
    