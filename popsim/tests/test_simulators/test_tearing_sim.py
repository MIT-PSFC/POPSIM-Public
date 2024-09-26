from popsim.simulators.tearing_sim.scenarios.simple_tearing import build_simple_tearing_sim_config
from popsim.simulate import simulate, SimInput



def test_tearing_sim():
    
    simulated_modes = [(2, 1), (3, 2)]
    reconstructed_modes = [1, 2, 3]

    tearing_sim, time_base, sim_initial_state, sim_params = build_simple_tearing_sim_config(simulated_modes, reconstructed_modes)

    sim_xarray = simulate(tearing_sim, SimInput(time=time_base, initial_state=sim_initial_state, params=sim_params), return_xarray=True)


    # Ensure the amplitudes of the simulated modes are large
    assert sim_xarray['output.lown_array_out.reconstructed_magnitudes.1'].max() > 1
    assert sim_xarray['output.lown_array_out.reconstructed_magnitudes.2'].max() > 1

    assert sim_xarray['output.rtnewspec_mirror_out.rms.1'].max() > 1
    assert sim_xarray['output.rtnewspec_mirror_out.rms.2'].max() > 1

    # Ensure the amplitudes of modes that are not simulated are small
    assert sim_xarray['output.lown_array_out.reconstructed_magnitudes.3'].max() < 1e-3
    assert sim_xarray['output.rtnewspec_mirror_out.rms.3'].max() < 1e-3

