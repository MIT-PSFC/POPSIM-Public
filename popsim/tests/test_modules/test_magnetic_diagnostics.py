import jax
import jax.numpy as jnp

import popsim.param_utils as param_utils
from popsim.modules.tearing import Tearing, DisruptionPhase, TearingPhase
from popsim.modules.magnetic_diagnostics import LowNArray, load_lown_config

def test_lown_array_same_amplitude_diff_phases():
    """Ensure the Low-N Array module returns the same amplitude for different phases of the same mode."""

    modes = [(2, 1), (3, 2)]

    dt = 1e-4 / 3  # s
    time_base = param_utils.make_time_base(t0=0.0, t1=3.0, dt=dt)
    config = Tearing.Config(
        magx_time=time_base,
        modes = modes
    )

    rot_dur = 1.0
    locking_dur = 0.2
    
    tearing_params = Tearing.Params(
        rot_dur=rot_dur,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=DisruptionPhase.NONE,
        tearing_phase=TearingPhase.ROTATING
    )

    tearing_module = Tearing(config=config)

    # Create a LowNArray module
    probe_connections, func_Bp_per_A = load_lown_config("popsim/data/tearing/")

    lown_array_config = LowNArray.Config(
        func_Bp_per_A=func_Bp_per_A,
        probe_connections=probe_connections,
        reconstructed_modes=[1,2,3]
    )

    lown_array_module = LowNArray(config=lown_array_config)

    test_phases = jnp.linspace(0, 2*jnp.pi, 10)

    test_tearing_states = [
        Tearing.State(W={mode: 50.0 for mode in modes},
                      F={mode: 1e3 for mode in modes},
                      mode_phase={mode: phase for mode in modes})
        for phase in test_phases
    ]

    resulting_magnitudes = []
    for tearing_state in test_tearing_states:
        _, tearing_output = tearing_module(tearing_state, tearing_params)
        lown_array_out = lown_array_module(tearing_output, tearing_module.config.modes)
        reconstructed_magnitudes = lown_array_out.reconstructed_magnitudes
        resulting_magnitudes.append(reconstructed_magnitudes)

    for mode in modes:
        toroidal_mode_number = mode[1]
        mode_magnitudes = jnp.asarray([all_magnitudes[toroidal_mode_number] for all_magnitudes in resulting_magnitudes])
        assert jnp.allclose(mode_magnitudes, jnp.ones_like(mode_magnitudes) * mode_magnitudes[0])

def test_lown_array_nonexistent_mode():
    """Ensure the measured amplitude of a nonexistent mode is very small."""

    modes = [(3, 2)]

    dt = 1e-4 / 3  # s
    time_base = param_utils.make_time_base(t0=0.0, t1=3.0, dt=dt)
    config = Tearing.Config(
        magx_time=time_base,
        modes = modes
    )

    rot_dur = 1.0
    locking_dur = 0.2
    
    tearing_params = Tearing.Params(
        rot_dur=rot_dur,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=DisruptionPhase.NONE,
        tearing_phase=TearingPhase.ROTATING
    )

    tearing_module = Tearing(config=config)

    # Create a LowNArray module
    probe_connections, func_Bp_per_A = load_lown_config("popsim/data/tearing/")

    lown_array_config = LowNArray.Config(
        func_Bp_per_A=func_Bp_per_A,
        probe_connections=probe_connections,
        reconstructed_modes=[1,2,3]
    )

    lown_array_module = LowNArray(config=lown_array_config)

    tearing_state = Tearing.State(W={mode: 50.0 for mode in modes},
                      F={mode: 1e3 for mode in modes},
                      mode_phase={mode: 0.0 for mode in modes})
    

    _, tearing_output = tearing_module(tearing_state, tearing_params)
    lown_array_out = lown_array_module(tearing_output, tearing_module.config.modes)
    reconstructed_magnitudes = lown_array_out.reconstructed_magnitudes

    assert reconstructed_magnitudes[1] < 1e-6*reconstructed_magnitudes[2]
    assert reconstructed_magnitudes[3] < 1e-6*reconstructed_magnitudes[2]
