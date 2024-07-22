from popsim import param_utils
from popsim.simulators.tearing_sim.model import TearingSim
from popsim.modules.tearing import Tearing, generate_disruption_phase_trajectory, generate_tearing_phase_trajectory
from popsim.modules.magnetic_diagnostics import LowNArray
from popsim.modules.magnetic_diagnostics import load_lown_config
from popsim.simulate import simulate

def test_tearing_sim():
    dt = 1e-4 / 3  # s
    time_base = param_utils.make_time_base(t0=0.0, t1=7.0, dt=dt)
    modes = [(2, 1), (3, 2)]
    tearing_config = Tearing.Config(
        magx_time=time_base,
        modes=modes,
    )

    tearing_initial_state = Tearing.State(
        W={mode: 0.0 for mode in modes},
        F={mode: 0.0 for mode in modes},
        mode_phase={mode: 0.0 for mode in modes}
    )

    rot_dur = 1.0
    locking_dur = 0.2
    trigger_time = 5.0
    disrupt_time = 6.5
    dur_tq_to_spike = 1e-3

    tearing_params = Tearing.Params(
        rot_dur=1.0,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=generate_disruption_phase_trajectory(
            disrupt_time, dur_tq_to_spike, time_base, dt
        ),
        tearing_phase=generate_tearing_phase_trajectory(
            trigger_time, rot_dur, locking_dur, time_base, dt
        ),
    )

    probe_connections, func_Bp_per_A = load_lown_config("popsim/data/tearing/")

    lown_array_config = LowNArray.Config(
        func_Bp_per_A=func_Bp_per_A,
        probe_connections=probe_connections,
        reconstructed_modes=[1,2,3]
    )

    sim_config = TearingSim.Config(
        tearing_config=tearing_config,
        lown_array_config=lown_array_config
    )

    sim_initial_state = TearingSim.State(
        tearing_state=tearing_initial_state
    )

    sim_params = TearingSim.Params(
        tearing_params=tearing_params
    )

    tearing_sim = TearingSim(config=sim_config)

    sim_xarray = simulate(tearing_sim, time_base, sim_initial_state, sim_params)