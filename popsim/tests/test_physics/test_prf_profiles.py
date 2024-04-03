
import jax.numpy as jnp

from popsim.physics.profiles import ProfileCalculator
from popsim.scenarios.sparc_prd.generic import Sparc2020TestData, load_cfspopcon_prd


def test_prf_profiles():
    input_parameters, *_= load_cfspopcon_prd()

    """
    Some numbers approximately correct for the SPARC PRD.
    """


    # calculate profiles
    pcalc = ProfileCalculator(profile_form=input_parameters["profile_form"], rho=jnp.linspace(0, 1, 100))

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

    # Simple checks on the outputs #TODO: make these more official
    assert profout["electron_density_profile"][0] > 40 and profout["electron_density_profile"][0] < 50
    assert profout["ion_density_profile"][0] > 30 and profout["ion_density_profile"][0] < 40
    assert profout["electron_temp_profile"][0] > 15 and profout["electron_temp_profile"][0] < 20
    assert profout["ion_temp_profile"][0] > 15 and profout["ion_temp_profile"][0] < 20
