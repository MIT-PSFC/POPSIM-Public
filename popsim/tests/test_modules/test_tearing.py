import popsim.param_utils as param_utils
from popsim.modules.tearing import (
    Tearing,
    ErrorFieldLocking,
    generate_disruption_phase_trajectory, 
    generate_tearing_phase_trajectory,
    load_active_circuit_overlaps,
    load_tf_overlap,
    calculate_total_overlap,
    calculate_locking_threshold,
    locked_mode_dynamics,
    DEFAULT_WDOT, 
    TQ_WDOT, 
    CQ_WDOT, 
    INITIAL_ROT_FREQ,
    TearingPhase,
)
from popsim.simulate import simulate, make_time_base, SimInput
from popsim import PACKAGE_ROOT
import pytest
import numpy as np

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


def test_load_tf_overlap():
    tf_overlap_file = f"{PACKAGE_ROOT}/data/tearing/error_field_sources/tfef.dat"

    # Ensure an error is raised if the percentile is out of range
    with pytest.raises(ValueError):
        load_tf_overlap(tf_overlap_file, -0.1)
    with pytest.raises(ValueError):
        load_tf_overlap(tf_overlap_file, 1.1)

    # Ensure the 99th percentile is greater than the 1st percentile
    tf_overlap_1 = load_tf_overlap(tf_overlap_file, 0.01)
    tf_overlap_2 = load_tf_overlap(tf_overlap_file, 0.99)

    assert tf_overlap_1 < tf_overlap_2


def test_calculate_total_overlap():
    # Make sure total overlap is calculated correctly in a simple case
    sample_config = ErrorFieldLocking.Config(
        tf_overlap = 10,
        static_source_overlaps={"microwave": 2, "fridge": 1},
        efc_efficiency=0.5,
    )

    sample_params = ErrorFieldLocking.Params(
        scaling_law_terms = {},
        scaling_law_params={},
        active_circuit_currents={"pf1": 3, "pf2": 4},
        active_circuit_overlaps={"pf1": 0.5, "pf2": 0.7},
        rational_surface_exists=1,
    )

    sample_module = ErrorFieldLocking(config=sample_config)

    _, out = sample_module(
        state = ErrorFieldLocking.State(tearing_phase=TearingPhase.NONE),
        params=sample_params
    )
    calculated_total = out["total_overlap"]

    expected_tf = 10
    expected_static = 3
    expected_active = (3 * 0.5) + (4 * 0.7)
    expected_total = (expected_tf + expected_static + expected_active) * 0.5

    assert np.isclose(calculated_total, expected_total)


def test_calculate_locking_threshold():
    ...


def test_locked_mode_dynamics():
    ...