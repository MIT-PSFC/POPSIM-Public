from popsim.modules.magnetic_diagnostics import LowNArray, load_lown_config
from popsim.modules.tearing import DEFAULT_WDOT, Tearing, generate_disruption_phase_trajectory, generate_tearing_phase_trajectory
from popsim.simulate import make_time_base
from popsim.simulators.tearing_sim.model import TearingSim


def build_simple_tearing_sim_config(simulated_modes: list[tuple[int, int]], reconstructed_modes: list[int]):
    """
    Build a simple tearing simulation configuration.

    Args:
        simulated_modes (list[tuple[int, int]]): The modes to simulate.
        reconstructed_modes (list[int]): The modes to be measured by the Low-N array.

    Returns:
        dict: The configuration.
    """

    for mode in simulated_modes:
        if mode not in DEFAULT_WDOT.keys():
            raise ValueError(f"Mode {mode} not presently supported by Tearing module.")

    # Define the time base.
    dt = 1e-4 / 3  # s
    time_base = make_time_base(t0=0.0, t1=7.0, dt=dt)

    # Define the tearing modes.
    tearing_config = Tearing.Config(
        modes=simulated_modes,
    )

    # Define the initial state.
    tearing_initial_state = Tearing.State(
        W={mode: 0.0 for mode in simulated_modes},
        F={mode: 0.0 for mode in simulated_modes},
        mode_phase={mode: 0.0 for mode in simulated_modes},
    )

    # Define the tearing parameters.
    rot_dur = 1.0
    locking_dur = 0.2
    trigger_time = 5.0
    disrupt_time = 6.5
    dur_tq_to_spike = 1e-3

    tearing_params = Tearing.Params(
        rot_dur=1.0,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=generate_disruption_phase_trajectory(disrupt_time, dur_tq_to_spike, time_base, dt),
        tearing_phase=generate_tearing_phase_trajectory(trigger_time, rot_dur, locking_dur, time_base, dt),
    )

    tearing_module = Tearing(config=tearing_config)

    # Define the Low-N array configuration.
    probe_connections, func_Bp_per_A = load_lown_config()

    lown_array_config = LowNArray.Config(
        func_Bp_per_A=func_Bp_per_A,
        probe_connections=probe_connections,
        reconstructed_modes=reconstructed_modes,
    )

    lown_array_module = LowNArray(config=lown_array_config)

    sim_config = TearingSim.Config(
        tearing_module=tearing_module,
        lown_array_module=lown_array_module,
    )

    sim_initial_state = TearingSim.State(tearing_state=tearing_initial_state)

    sim_params = TearingSim.Params(tearing_params=tearing_params)

    sim_model = TearingSim(config=sim_config)

    return sim_model, time_base, sim_initial_state, sim_params
