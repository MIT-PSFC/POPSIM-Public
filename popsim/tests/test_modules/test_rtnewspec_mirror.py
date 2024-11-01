import jax.numpy as jnp

from popsim.modules.rtnewspec_mirror import RTNewSpecMirror

def test_rtnewspec_mirror_call():
    probe_offset = 0.5

    config = RTNewSpecMirror.Config(
        d_theta=jnp.rad2deg(probe_offset),
        probe1_id="probe1",
        probe2_id="probe2",
        nsamples=2048,
        alpha=0.001, # TODO(ZanderKeith): This is some factor that needs to be included to get [T] out of the FFT.
        f_probe=60e3,
        nsmth=3,
        max_modes=3,
    )

    rtnewspec_mirror = RTNewSpecMirror(config=config)

    # Make repeating sinusoidal signals with 10 kHz frequency as if they were being sampled at 60 kHz
    time = jnp.arange(2048)/60e3
    probe1_signal = jnp.sin(2*jnp.pi*10e3*time)
    probe2_signal = jnp.sin(2*jnp.pi*10e3*time - probe_offset)

    # Call the module with the signals fora while and check the output at the end
    present_state = RTNewSpecMirror.State()

    for i in range(1000):
        params = RTNewSpecMirror.Params(
            probe1_signal=probe1_signal[i],
            probe2_signal=probe2_signal[i],
        )
        state, output = rtnewspec_mirror(present_state, params)
        present_state = state
    
    # Check that the output is large for n=1 but small for n=2
    assert output.rms[1] > 1.0
    assert output.rms[2] < 0.1

def test_calculated_rms_simple():
    """Test the RMS calculation of a simple sinusoidal signal with known phase offset."""

    probe_offset = jnp.pi/5

    config = RTNewSpecMirror.Config(
        d_theta=jnp.rad2deg(probe_offset),
        probe1_id="probe1",
        probe2_id="probe2",
        nsamples=2048,
        alpha=0.001, # TODO(ZanderKeith): This is some factor that needs to be included to get [T] out of the FFT.
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
    assert rms_values[1] > 1.0
    assert rms_values[2] < 0.1

def test_calculated_rms_inverted_phase():
    """Test the RMS calculation of a simple sinusoidal signal with known phase offset, but inverted."""

    probe_offset = jnp.pi/5

    config = RTNewSpecMirror.Config(
        d_theta=jnp.rad2deg(probe_offset),
        probe1_id="probe1",
        probe2_id="probe2",
        nsamples=2048,
        alpha=0.001, # TODO(ZanderKeith): This is some factor that needs to be included to get [T] out of the FFT.
        f_probe=60e3,
        nsmth=3,
        max_modes=2,
    )

    rtnewspec_mirror = RTNewSpecMirror(config=config)

    # Make repeating sinusoidal signals with 10 kHz frequency as if they were being sampled at 60 kHz
    time = jnp.arange(2048)/60e3
    probe1_signal = jnp.sin(2*jnp.pi*10e3*time)
    probe2_signal = jnp.sin(2*jnp.pi*10e3*time + probe_offset)

    # Calculate the RMS values directly
    rms_values = rtnewspec_mirror.calculate_rms(probe1_signal, probe2_signal)

    # Check that the output is large for n=1 but small for n=2
    assert rms_values[1] > 1.0
    assert rms_values[2] < 0.1

def test_calculated_rms_separate_negative_n():
    """Test the reconstruction of a sinusoidal signal with two modes, one positive and one negative."""

    probe_offset = jnp.pi/5

    config = RTNewSpecMirror.Config(
        d_theta=jnp.rad2deg(probe_offset),
        probe1_id="probe1",
        probe2_id="probe2",
        nsamples=2048,
        alpha=0.001, # TODO(ZanderKeith): This is some factor that needs to be included to get [T] out of the FFT.
        f_probe=60e3,
        nsmth=3,
        max_modes=2,
        separate_negative_n=True,
    )

    rtnewspec_mirror = RTNewSpecMirror(config=config)

    # Make repeating sinusoidal signals with 10 kHz frequency as if they were being sampled at 60 kHz
    # Here we have an n=1 mode and a n=-2 mode
    time = jnp.arange(2048)/60e3
    probe1_n1_signal = 0.5*jnp.sin(2*jnp.pi*10e3*time)
    probe2_n1_signal = 0.5*jnp.sin(2*jnp.pi*10e3*time - probe_offset)
    probe1_n2_signal = 0.5*jnp.sin(2*(2*jnp.pi*10e3*time))
    probe2_n2_signal = 0.5*jnp.sin(2*(2*jnp.pi*10e3*time + probe_offset))

    probe1_signal = probe1_n1_signal + probe1_n2_signal
    probe2_signal = probe2_n1_signal + probe2_n2_signal

    # Calculate the RMS values directly
    rms_values = rtnewspec_mirror.calculate_rms(probe1_signal, probe2_signal)

    # Check that the output is large for n=1 and n=-2
    # but small for n=2 and n=-1
    assert rms_values[1] > 1.0
    assert rms_values[-2] > 1.0
    
    assert rms_values[-1] < 0.1
    assert rms_values[2] < 0.1