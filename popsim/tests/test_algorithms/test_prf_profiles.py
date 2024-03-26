
import jax.numpy as jnp
import xarray as xr

from popsim.tests import load_sparc_prd_data, Sparc2020Data
from popsim.algorithms.profiles import ProfileCalculator


def test_prf_profiles(load_sparc_prd_data):
    input_parameters, algorithm, points, impurity_types, impurity_concentrations = load_sparc_prd_data

    """
    Some numbers approximately correct for the SPARC PRD.
    """


    # calculate profiles
    pcalc = ProfileCalculator(profile_form=input_parameters['profile_form'], rho=jnp.linspace(0, 1, 100))

    """
    Prepare profile inputs
    """
    average_electron_density_19 = Sparc2020Data.ne_vol
    average_electron_temp_keV = Sparc2020Data.Te_vol
    average_ion_temp_keV = Sparc2020Data.Ti_vol
    ion_density_peaking_offset = input_parameters['ion_density_peaking_offset'].magnitude
    electron_density_peaking_offset = input_parameters['electron_density_peaking_offset'].magnitude
    temperature_peaking = input_parameters['temperature_peaking'].magnitude
    major_radius = input_parameters['major_radius'].magnitude
    z_effective = Sparc2020Data.Zeff
    dilution = Sparc2020Data.dilution
    normalized_inverse_temp_scale_length = input_parameters['normalized_inverse_temp_scale_length'].magnitude

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
        beta_toroidal=Sparc2020Data.beta, # beta is dominated by the toroidal component.
        normalized_inverse_temp_scale_length=normalized_inverse_temp_scale_length,
    )

    # Simple checks on the outputs #TODO: make these more official
    assert profout["electron_density_profile"][0] > 40 and profout["electron_density_profile"][0] < 50
    assert profout["ion_density_profile"][0] > 30 and profout["ion_density_profile"][0] < 40
    assert profout["electron_temp_profile"][0] > 15 and profout["electron_temp_profile"][0] < 20
    assert profout["ion_temp_profile"][0] > 15 and profout["ion_temp_profile"][0] < 20
