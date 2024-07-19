import diffrax
import jax
import jax.numpy as jnp
from popsim.modules.hmode_dynamics import HmodeDynamics
from popsim.simulate import simulate

import popsim.param_utils as param_utils
from popsim.modules.tearing import Tearing, generate_disruption_phase_trajectory, generate_tearing_phase_trajectory, DEFAULT_WDOT, TQ_WDOT, CQ_WDOT, INITIAL_ROT_FREQ
from popsim.simulate import simulate

def test_mode_growth_and_freq():
    """Ensure the simulated mode growth and rotation frequency match the hard-coded values"""
    dt = 1e-4 / 3  # s
    time_base = param_utils.make_time_base(t0=0.0, t1=3.0, dt=dt)
    modes = [(2, 1), (3, 2)]
    config = Tearing.Config(
        magx_time=time_base,
        modes = modes
    )

    W = {mode: 0.0 for mode in modes}
    F = {mode: 0.0 for mode in modes}
    mode_phase={mode: 0.0 for mode in modes}

    initial_state = Tearing.State(W=W, F=F, mode_phase=mode_phase)

    trigger_time = 0.5
    rot_dur = 1.0
    locking_dur = 0.2
    survival_time = 0.3
    disrupt_time = trigger_time + rot_dur + locking_dur + survival_time
    dur_tq_to_spike = 1e-3

    lock_time = trigger_time + rot_dur + locking_dur
    cq_time = disrupt_time + dur_tq_to_spike
    
    params = Tearing.Params(
        rot_dur=rot_dur,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=generate_disruption_phase_trajectory(
            disrupt_time, dur_tq_to_spike, time_base, dt
        ),
        tearing_phase=generate_tearing_phase_trajectory(
            trigger_time, rot_dur, locking_dur, time_base, dt
        ),
    )

    jax.config.update("jax_platforms", "cpu")
    tearing_module = Tearing(config=config)

    sol_xarray = simulate(tearing_module, time_base, initial_state, params)

    for mode in modes:
        W_dot = sol_xarray[f"aux.state_dot.W.{str(mode)}"]
        assert W_dot.sel(time=0, method="nearest") == 0.0
        assert W_dot.sel(time=trigger_time, method="nearest") == DEFAULT_WDOT[mode]
        assert W_dot.sel(time=trigger_time+rot_dur, method="nearest") == DEFAULT_WDOT[mode]
        assert W_dot.sel(time=lock_time, method="nearest") == DEFAULT_WDOT[mode]
        assert W_dot.sel(time=disrupt_time, method="nearest") == TQ_WDOT[mode]
        assert W_dot.sel(time=cq_time, method="nearest") == CQ_WDOT[mode]

        current = sol_xarray[f"aux.mode_current.{str(mode)}"]
        assert current.sel(time=0, method="nearest") == 0.0
        assert current.sel(time=trigger_time, method="nearest") > 0.0
        assert current.sel(time=cq_time+1e3*dt, method="nearest") == 0.0

        F_dot = sol_xarray[f"aux.state_dot.F.{str(mode)}"]
        assert F_dot.sel(time=0, method="nearest") == 0.0
        assert F_dot.sel(time=trigger_time+dt, method="nearest") < 0.0
        assert F_dot.sel(time=lock_time, method="nearest") == 0.0

        F = sol_xarray[f"aux.mode_freq.{str(mode)}"]
        assert F.sel(time=0, method="nearest") == 0.0
        assert F.sel(time=trigger_time, method="nearest") == INITIAL_ROT_FREQ[mode]
        assert F.sel(time=lock_time, method="nearest") == 0.0
