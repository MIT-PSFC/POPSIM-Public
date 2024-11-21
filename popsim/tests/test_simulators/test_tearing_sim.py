from popsim.simulators.tearing_sim.scenarios.simple_tearing import build_simple_tearing_sim_config
from popsim.simulate import simulate, SimInput
from popsim.modules.tearing import Tearing, generate_disruption_phase_trajectory, generate_tearing_phase_trajectory
from popsim.simulate import make_time_base
from popsim.modules.magnetic_diagnostics import LowNArray, load_lown_config
from popsim.modules.rtnewspec_mirror import RTNewSpecMirror
from popsim.modules.magnetic_diagnostics import BFieldPoloidalProbes
from popsim.simulators.tearing_sim.model import TearingSim



def test_tearing_sim():
    
    simulated_modes = [(2, 1), (3, 2)]
    reconstructed_modes = [1, 2, 3]

    tearing_sim, time_base, sim_initial_state, sim_params = build_simple_tearing_sim_config(simulated_modes, reconstructed_modes)

    sim_xarray = simulate(tearing_sim, SimInput(time=time_base, initial_state=sim_initial_state, params=sim_params), return_xarray=True, record_state=False)

    # Ensure the amplitudes of the simulated modes are large
    assert sim_xarray['output.lown_array_out.reconstructed_magnitudes.1'].max() > 1
    assert sim_xarray['output.lown_array_out.reconstructed_magnitudes.2'].max() > 1

    assert sim_xarray['output.rtnewspec_mirror_out.rms.1'].mean() > 1e8
    assert sim_xarray['output.rtnewspec_mirror_out.rms.2'].mean() > 1e8

    # Ensure the amplitudes of modes that are not simulated are small
    assert sim_xarray['output.lown_array_out.reconstructed_magnitudes.3'].max() < 1e-3
    assert sim_xarray['output.rtnewspec_mirror_out.rms.3'].mean() < 1e8

def test_tearing_sim_notebook():
    modes = [(2, 1), (3, 2)]
    tearing_config = Tearing.Config(
        modes=modes,
    )

    tearing_initial_state = Tearing.State(
        W={mode: 0.0 for mode in modes}, F={mode: 0.0 for mode in modes}, mode_phase={mode: 0.0 for mode in modes}
    )

    dt = 1 / 60e3  # s
    time_base = make_time_base(t0=0.0, t1=2.0, dt=dt)

    rot_dur = 1.0
    locking_dur = 0.2
    trigger_time = 0.1
    disrupt_time = 1.2
    dur_tq_to_spike = 1e-3

    tearing_params = Tearing.Params(
        rot_dur=rot_dur,  # s
        locking_dur=locking_dur,  # s
        disruption_phase=generate_disruption_phase_trajectory(disrupt_time, dur_tq_to_spike, time_base, dt),
        tearing_phase=generate_tearing_phase_trajectory(trigger_time, rot_dur, locking_dur, time_base, dt),
    )

    tearing_module = Tearing(config=tearing_config)

    probe_connections, func_Bp_per_A = load_lown_config()

    lown_array_config = LowNArray.Config(func_Bp_per_A=func_Bp_per_A, probe_connections=probe_connections, reconstructed_modes=[1, 2, 3])

    lown_array_module = LowNArray(config=lown_array_config)

    rtnewspec_mirror_module_config = RTNewSpecMirror.Config(
        d_theta=20,
        probe1_id="sample_probe_1_identifier",  # Probe id's need to match the
        probe2_id="sample_probe_2_identifier",  # id's in the BFieldPoloidalProbes module
        nsamples=2048,
        alpha=0.001,
        f_probe=30e3,
        nsmth=3,
        max_modes=3,
    )

    rtnewspec_mirror_module = RTNewSpecMirror(config=rtnewspec_mirror_module_config)

    sim_config = TearingSim.Config(
    tearing_module=tearing_module,
    lown_array_module=lown_array_module,
    rtnewspec_mirror_module=rtnewspec_mirror_module,
    b_field_poloidal_probes_module=BFieldPoloidalProbes.default_setup(),
)

    sim_initial_state = TearingSim.State(tearing_state=tearing_initial_state, rtnewspec_mirror_state=RTNewSpecMirror.State())

    sim_params = TearingSim.Params(tearing_params=tearing_params)

    tearing_sim = TearingSim(config=sim_config)

    sim_input = SimInput(time=time_base, initial_state=sim_initial_state, params=sim_params)

    #with ipdb.launch_ipdb_on_exception():
    #    sim_xarray = simulate(tearing_sim, sim_inputs=sim_input)

    sim_xarray = simulate(tearing_sim, sim_inputs=sim_input)


            
