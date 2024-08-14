import jax
import jax.numpy as jnp

import popsim.simulators.comet_mirror.scenarios.sparc_prd as sparc_prd_cm
from popsim.simulators.scenario_data.sparc_prd import Sparc2020TestData
from popsim.simulate import simulate, SimInput


def generate_sim_and_checks(only_return_final: bool = True):
    # Test that the simulator runs and compare against a reference solution.
    model, state, params = sparc_prd_cm.build_comet_mirror_config()
    ts = jnp.linspace(0, 1.0, 10)
    sol = simulate(model, SimInput(time=ts, initial_state=state, params=params), return_xarray=False)

    if only_return_final:
        out = jax.tree.map(lambda x: x[-1], sol)
    else:
        out = sol
    out_state, out_aux = out["state"], out["output"]["aux_data"]

    # The checks take the form of (actual, expected, percent_tolerance).
    PsepB0R0 = (out_aux["P_tau_MW"] * params.magnetic_field_on_axis)/params.geometry.major_radius

    checks = {
        "P_fusion_MW": (out_aux["P_fusion_MW"], Sparc2020TestData.Pfusion, 20.0),
        "P_ohmic_MW": (out_aux["P_ohmic_MW"], Sparc2020TestData.Pohm, 20.0),
        "P_aux_MW": (out_aux["params"]["P_aux_MW"], Sparc2020TestData.Paux, 1e-3), # Should be exact because it's a parameter.
        "beta_t": (out_aux["beta_t"], Sparc2020TestData.beta, 20.0),
        "tau_E": (out_aux["tau_E"], Sparc2020TestData.tauE, 10.0),
        "P_rad_MW": (out_aux["P_rad_MW"], Sparc2020TestData.Prad, 15.0),
        "ion_density": (out_state.density_state.total_volume_average_ion_density, Sparc2020TestData.ni_vol, 15.0),
        "average_electron_density_19": (out_aux["average_electron_density_19"], Sparc2020TestData.ne_vol, 5.0),
        "average_electron_temp_keV": (out_aux["average_electron_temp_keV"], Sparc2020TestData.Te_vol, 10.0),
        "average_ion_temp_keV": (out_aux["average_ion_temp_keV"], Sparc2020TestData.Ti_vol, 10.0),
        "q_star": (out_aux["q_star"], Sparc2020TestData.qstar_uckan, 10.0),
        "dilution": (out_aux["dilution"], Sparc2020TestData.dilution, 20.0),
        "z_effective": (out_aux["z_effective"], Sparc2020TestData.Zeff, 20.0),
        "PsepB0R0": (PsepB0R0, Sparc2020TestData.PsepB0R0, 20.0),
    }

    def tuple_to_dict(tup):
        out = {
            "sim_result": tup[0],
            "creely2020": tup[1],
            "percent_tolerance": tup[2]
        }
        return out

    checks = {key: tuple_to_dict(val) for key, val in checks.items()}

    if only_return_final:
        return checks
    else:
        return checks, sol, ts


def test_comet_mirror_prd():

    def percent_error(expected, actual):
        return 100 * jnp.abs(expected - actual) / expected

    checks = generate_sim_and_checks(only_return_final=True)

    percent_errors = {}
    for key, check_case in checks.items():
        percent_errors[key] = percent_error(check_case["creely2020"], check_case["sim_result"])
    for key, percent_error in percent_errors.items():
        tol = checks[key]["percent_tolerance"]
        assert percent_error < tol, f"{key} has a percent error of {percent_error} which is greater than the tolerance of {tol}."



def test_comet_mirror_configuration_modes():
    pass