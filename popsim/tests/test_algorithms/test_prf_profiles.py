
import jax.numpy as jnp
import xarray as xr

from popsim.tests import load_sparc_prd_data
from popsim.algorithms.profiles import ProfileCalculator
from popsim.algorithms.zeff_and_dilution_from_impurities import CalcZeffAndDilutionFromImpurities, TempDensImpurities


def test_prf_profiles(load_sparc_prd_data):
    input_parameters, algorithm, points, impurity_types, impurity_concentrations = load_sparc_prd_data


    # get dilution
    impcalc = CalcZeffAndDilutionFromImpurities(impurity_types)

    """
    Some numbers approximately correct for the SPARC PRD.
    """
    beta = 0.012
    Bt = 12.2
    # Use equation 4.4 from Creely 2020.
    # This pressure is in keV * 1e19 m^-3.
    average_pressure = (beta * Bt**2.0) / 4.02e-3
    average_ion_density = 27.0 # 1e19 m^-3
    td_calc = TempDensImpurities(zeff_and_dilution_calc=impcalc)
    out = td_calc(average_pressure=average_pressure,
                          average_ion_density=average_ion_density,
                          ion_to_electron_temp_ratio=1.0,
                          impurity_concentrations=impurity_concentrations)


    # calculate profiles
    pcalc = ProfileCalculator(profile_form=input_parameters['profile_form'])

    """
    Prepare profile inputs
    """
    average_electron_density_19 = float(out['average_electron_density'])
    average_electron_temp_keV = float(out['average_electron_temp'])
    average_ion_temp_keV = float(out['average_ion_temp'])
    ion_density_peaking_offset = 1.0 * input_parameters['ion_density_peaking_offset'].magnitude
    electron_density_peaking_offset = 1.0 * input_parameters['electron_density_peaking_offset'].magnitude
    temperature_peaking = 1.0 * input_parameters['temperature_peaking'].magnitude
    major_radius = 1.0 * input_parameters['major_radius'].magnitude
    z_effective = float(out['z_effective'])
    dilution = float(out['dilution'])
    beta_toroidal = beta
    normalized_inverse_temp_scale_length = 1.0 * input_parameters['normalized_inverse_temp_scale_length'].magnitude

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
        beta_toroidal=beta_toroidal,
        normalized_inverse_temp_scale_length=normalized_inverse_temp_scale_length,
    )

    # Simple checks on the outputs #TODO: make these more official
    assert profout["electron_density_profile"][0] > 40 and profout["electron_density_profile"][0] < 50
    assert profout["ion_density_profile"][0] > 30 and profout["ion_density_profile"][0] < 40
    assert profout["electron_temp_profile"][0] > 15 and profout["electron_temp_profile"][0] < 20
    assert profout["ion_temp_profile"][0] > 15 and profout["ion_temp_profile"][0] < 20
