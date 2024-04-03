import jax
import jax.numpy as jnp

import popsim.enums as penums
import popsim.scenarios.sparc_prd.comet_mirror as sparc_prd_cm
from popsim.scenarios.sparc_prd.generic import Sparc2020TestData
from popsim.simulators.comet_mirror.simulate import simulate


def test_comet_mirror_prd():
    # Test that the simulator runs and compare against a reference solution.
    model, state, params = sparc_prd_cm.build_comet_mirror_config()
    ts = jnp.linspace(0, 1.0, 10)
    sol = simulate(model, ts, state, params, return_xarray=False)
    out = jax.tree_map(lambda x: x[-1], sol.ys)
    final_state, final_aux = out["state"], out["aux"]

    def percent_error(expected, actual):
        return 100 * jnp.abs(expected - actual) / expected

    # The checks take the form of (actual, expected, percent_tolerance).
    PsepB0R0 = (final_aux["P_tau_MW"] * params.magnetic_field_on_axis)/params.geometry.major_radius
    checks = {
        "P_fusion_MW": (final_aux["P_fusion_MW"], Sparc2020TestData.Pfusion, 20.0),
        "P_ohmic_MW": (final_aux["P_ohmic_MW"], Sparc2020TestData.Pohm, 20.0),
        "P_aux_MW": (final_aux["params"]["P_aux_MW"], Sparc2020TestData.Paux, 1e-3), # Should be exact because it's a parameter.
        "beta_t": (final_aux["beta_t"], Sparc2020TestData.beta, 20.0),
        "tau_E": (final_aux["tau_E"], Sparc2020TestData.tauE, 10.0),
        "P_rad_MW": (final_aux["P_rad_MW"], Sparc2020TestData.Prad, 15.0),
        "ion_density": (final_state.density_state.total_volume_average_ion_density, Sparc2020TestData.ni_vol, 15.0),
        "average_electron_density_19": (final_aux["average_electron_density_19"], Sparc2020TestData.ne_vol, 5.0),
        "average_electron_temp_keV": (final_aux["average_electron_temp_keV"], Sparc2020TestData.Te_vol, 10.0),
        "average_ion_temp_keV": (final_aux["average_ion_temp_keV"], Sparc2020TestData.Ti_vol, 10.0),
        "q_star": (final_aux["q_star"], Sparc2020TestData.qstar_uckan, 10.0),
        "dilution": (final_aux["dilution"], Sparc2020TestData.dilution, 20.0),
        "z_effective": (final_aux["z_effective"], Sparc2020TestData.Zeff, 20.0),
        "PsepB0R0": (PsepB0R0, Sparc2020TestData.PsepB0R0, 20.0),
    }

    percent_errors = {}
    for key, (actual, expected, tol) in checks.items():
        percent_errors[key] = percent_error(expected, actual)
    for key, percent_error in percent_errors.items():
        assert percent_error < checks[key][2], f"{key} has a percent error of {percent_error} which is greater than the tolerance of {checks[key][2]}."



def test_comet_mirror_configuration_modes():
    model, state, params = sparc_prd_cm.build_comet_mirror_config()

    params.fueling19[penums.Impurity.Tungsten] = {0.0: 0.0, 1.0: 0.1, 1.1: 0.0}
