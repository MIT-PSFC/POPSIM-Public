
import jax.numpy as jnp

from popsim.physics.profiles import ProfileCalculator
from popsim.simulators.scenario_data.sparc_prd import Sparc2020TestData, load_cfspopcon_prd
from popsim.interfaces.sparc_public import load_prd_transp_profiles


def test_prf_profiles():
    input_parameters, *_= load_cfspopcon_prd()
    transp_profiles = load_prd_transp_profiles()

    # calculate profiles
    pcalc = ProfileCalculator(density_profile_form=input_parameters["density_profile_form"], temp_profile_form=input_parameters["temp_profile_form"], rho=transp_profiles.rho.values)

    """
    Prepare profile inputs
    """
    average_electron_density_19 = Sparc2020TestData.ne_vol
    average_electron_temp_keV = Sparc2020TestData.Te_vol
    average_ion_temp_keV = Sparc2020TestData.Ti_vol
    ion_density_peaking_offset = input_parameters["ion_density_peaking_offset"]
    electron_density_peaking_offset = input_parameters["electron_density_peaking_offset"]
    temperature_peaking = input_parameters["temperature_peaking"]
    major_radius = input_parameters["major_radius"]
    z_effective = Sparc2020TestData.Zeff
    dilution = Sparc2020TestData.dilution
    normalized_inverse_temp_scale_length = input_parameters["normalized_inverse_temp_scale_length"]

    profout = pcalc(
        average_electron_density_19=average_electron_density_19,
        average_electron_temp_keV=average_electron_temp_keV,
        average_ion_temp_keV=average_ion_temp_keV,
        ion_density_peaking_offset=ion_density_peaking_offset,
        electron_density_peaking_offset=electron_density_peaking_offset,
        temperature_peaking=temperature_peaking,
        major_radius=major_radius,
        z_effective=z_effective,
        dilution=dilution,
        beta_toroidal=Sparc2020TestData.beta, # beta is dominated by the toroidal component.
        normalized_inverse_temp_scale_length=normalized_inverse_temp_scale_length,
    )

    ne20 = 0.1 * profout["electron_density_profile"].values
    te_keV = profout["electron_temp_profile"].values
    ti_keV = profout["ion_temp_profile"].values
    ne20_transp = transp_profiles.ne20.values
    te_keV_transp = transp_profiles.Te_keV.values
    ti_keV_transp = transp_profiles.Ti_keV.values

    def calc_percent_error(a, b):
        return jnp.abs(a - b) / a * 100

    # Check at the axis.
    PERCENT_ERROR = 25.0
    assert calc_percent_error(ne20[0], ne20_transp[0]) < PERCENT_ERROR
    assert calc_percent_error(te_keV[0], te_keV_transp[0]) < PERCENT_ERROR
    assert calc_percent_error(ti_keV[0], ti_keV_transp[0]) < PERCENT_ERROR

    # Check at rho=0.5.
    assert calc_percent_error(ne20[50], ne20_transp[50]) < PERCENT_ERROR
    assert calc_percent_error(te_keV[50], te_keV_transp[50]) < PERCENT_ERROR
    assert calc_percent_error(ti_keV[50], ti_keV_transp[50]) < PERCENT_ERROR

    # Check at rho=0.8.
    assert calc_percent_error(ne20[80], ne20_transp[80]) < PERCENT_ERROR
    assert calc_percent_error(te_keV[80], te_keV_transp[80]) < PERCENT_ERROR
    assert calc_percent_error(ti_keV[80], ti_keV_transp[80]) < PERCENT_ERROR