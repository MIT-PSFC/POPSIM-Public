import popsim.param_utils as param_utils
from popsim.modules.tearing import Tearing, generate_disruption_phase_trajectory, generate_tearing_phase_trajectory, DEFAULT_WDOT, TQ_WDOT, CQ_WDOT, INITIAL_ROT_FREQ
from popsim.simulate import simulate, make_time_base, SimInput
import pytest

def run_tearing_test_sim():
    """Ensure the simulated mode growth and rotation frequency match the hard-coded values"""
    dt = 1e-4 / 3  # s
    time_base = make_time_base(t0=0.0, t1=3.0, dt=dt)
    modes = [(2, 1), (3, 2)]
    config = Tearing.Config(
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

    tearing_module = Tearing(config=config)

    sol_xarray = simulate(tearing_module, SimInput(time=time_base, initial_state=initial_state, params=params), return_xarray=True)
    aux_data = locals()
    return sol_xarray, aux_data

@pytest.fixture()
def tearing_test_sim():
    return run_tearing_test_sim()

def test_mode_growth_and_freq(tearing_test_sim):
    sol_xarray, aux_data = tearing_test_sim

    for mode in aux_data["modes"]:
        W_dot = sol_xarray[f"output.state_dot.W.{str(mode)}"]
        assert W_dot.sel(time=0, method="nearest") == 0.0
        assert W_dot.sel(time=aux_data["trigger_time"], method="nearest") == DEFAULT_WDOT[mode]
        assert W_dot.sel(time=aux_data["trigger_time"]+aux_data["rot_dur"], method="nearest") == DEFAULT_WDOT[mode]
        assert W_dot.sel(time=aux_data["lock_time"], method="nearest") == DEFAULT_WDOT[mode]
        assert W_dot.sel(time=aux_data["disrupt_time"], method="nearest") == TQ_WDOT[mode]
        assert W_dot.sel(time=aux_data["cq_time"], method="nearest") == CQ_WDOT[mode]

        current = sol_xarray[f"output.mode_current.{str(mode)}"]
        assert current.sel(time=0, method="nearest") == 0.0
        assert current.sel(time=aux_data["trigger_time"] + aux_data["dt"], method="nearest") > 0.0
        assert current.sel(time=aux_data["cq_time"]+1e3*aux_data["dt"], method="nearest") == 0.0

        F_dot = sol_xarray[f"output.state_dot.F.{str(mode)}"]
        assert F_dot.sel(time=0, method="nearest") == 0.0
        assert F_dot.sel(time=aux_data["trigger_time"]+aux_data["dt"], method="nearest") < 0.0
        assert F_dot.sel(time=aux_data["lock_time"], method="nearest") == 0.0

        F = sol_xarray[f"output.mode_freq.{str(mode)}"]
        assert F.sel(time=0, method="nearest") == 0.0
        assert F.sel(time=aux_data["trigger_time"], method="nearest") == INITIAL_ROT_FREQ[mode]
        assert F.sel(time=aux_data["lock_time"], method="nearest") == 0.0
