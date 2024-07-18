import jax
import os

import popsim.param_utils as param_utils
from popsim.modules.tearing import Island, Tearing, DisruptionPhase, TearingPhase
from popsim.simulate import simulate
from popsim.modules.magnetic_diagnostics import LowNArray, load_lown_config

def test_lown_array_same_amplitude_diff_phases():
    jax.config.update("jax_platforms", "cpu")

    dt = 1e-4 / 3  # s
    time_base = param_utils.make_time_base(t0=0.0, t1=3.0, dt=dt)
    config = Tearing.Config(
        magx_time=time_base
    )

    islands = [Island(2, 1), Island(3, 2)]

    rot_dur = 1.0
    locking_dur = 0.2
    
    tearing_params = Tearing.Params(
        rot_dur=rot_dur,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=DisruptionPhase.NONE,
        tearing_phase=TearingPhase.ROTATING
    )

    tearing_module = Tearing(config=config, islands=islands)

    # Create a LowNArray module
    probe_connections, func_Bp_per_A = load_lown_config("popsim/data/tearing/")

    lown_array_config = LowNArray.Config(
        func_Bp_per_A=func_Bp_per_A,
        probe_connections=probe_connections,
        reconstructed_modes=[1,2,3]
    )

    lown_array_module = LowNArray(config=lown_array_config)

    W = {island: 0.0 for island in islands}
    F = {island: 0.0 for island in islands}
    mode_phase={island: 0.0 for island in islands}

    tearing_state = Tearing.State(W=W, F=F, mode_phase=mode_phase)

    # Get the measured output from a given Tearing module output
    _, tearing_output = tearing_module(tearing_state, tearing_params)

    # Get the LowNArray module's output
    lown_array_out = lown_array_module(tearing_output, tearing_module.config.islands)
    reconstructed_magnitudes = lown_array_out.reconstructed_magnitudes

    # Get the Tearing module's output
    print("beans")