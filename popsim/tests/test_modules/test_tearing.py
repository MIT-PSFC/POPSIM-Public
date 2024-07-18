import diffrax
import jax
import jax.numpy as jnp
from popsim.modules.hmode_dynamics import HmodeDynamics
from popsim.simulate import simulate

import popsim.param_utils as param_utils
from popsim.modules.tearing import Island, Tearing, generate_disruption_phase_trajectory, generate_tearing_phase_trajectory
from popsim.simulate import simulate

def test_island_comparison():
    island1 = Island(2, 1)
    island2 = Island(2, 1)
    assert island1 == island2

    island3 = Island(3, 2)
    assert island1 > island3
    assert island3 < island1

def test_island_hash():
    island1 = Island(2, 1)
    island2 = Island(2, 1)
    island3 = Island(3, 2)

    assert hash(island1) == hash(island2)
    assert hash(island1) != hash(island3)

def test_mode_growth_and_freq():
    """Ensure the simulated mode growth and rotation frequency match the hard-coded values"""
    dt = 1e-4 / 3  # s
    time_base = param_utils.make_time_base(t0=0.0, t1=3.0, dt=dt)
    config = Tearing.Config(
        magx_time=time_base
    )

    islands = [Island(2, 1), Island(3, 2)]

    W = {island: 0.0 for island in islands}
    F = {island: 0.0 for island in islands}
    mode_phase={island: 0.0 for island in islands}

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
    tearing_module = Tearing(config=config, islands=islands)

    sol_xarray = simulate(tearing_module, time_base, initial_state, params)

    for island in islands:
        island_tuple = (island.m, island.n)

        W_dot = sol_xarray[f"aux.state_dot.W.{str(island)}"]
        assert W_dot.sel(time=0, method="nearest") == 0.0
        assert W_dot.sel(time=trigger_time, method="nearest") == Island._DEFAULT_WDOT_DICT[island_tuple]
        assert W_dot.sel(time=trigger_time+rot_dur, method="nearest") == Island._DEFAULT_WDOT_DICT[island_tuple]
        assert W_dot.sel(time=lock_time, method="nearest") == Island._DEFAULT_WDOT_DICT[island_tuple]
        assert W_dot.sel(time=disrupt_time, method="nearest") == Island._TQ_WDOT_DICT[island_tuple]
        assert W_dot.sel(time=cq_time, method="nearest") == Island._CQ_WDOT_DICT[island_tuple]

        current = sol_xarray[f"aux.mode_current.{str(island)}"]
        assert current.sel(time=0, method="nearest") == 0.0
        assert current.sel(time=trigger_time, method="nearest") > 0.0
        assert current.sel(time=cq_time+1e3*dt, method="nearest") == 0.0

        F_dot = sol_xarray[f"aux.state_dot.F.{str(island)}"]
        assert F_dot.sel(time=0, method="nearest") == 0.0
        assert F_dot.sel(time=trigger_time+dt, method="nearest") < 0.0
        assert F_dot.sel(time=lock_time, method="nearest") == 0.0

        F = sol_xarray[f"aux.mode_freq.{str(island)}"]
        assert F.sel(time=0, method="nearest") == 0.0
        assert F.sel(time=trigger_time, method="nearest") == Island._INITIAL_ROT_FREQ_DICT[island_tuple]
        assert F.sel(time=lock_time, method="nearest") == 0.0
