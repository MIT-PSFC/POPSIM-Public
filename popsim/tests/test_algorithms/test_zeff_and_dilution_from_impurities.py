import jax
import jax.numpy as jnp
import pytest

from popsim.algorithms.zeff_and_dilution_from_impurities import (
    CalcZeffAndDilutionFromImpurities, ImpurityCalculator, TempDensImpurities)
from popsim.enums import Impurity
from popsim.interfaces.atomic_data import read_atomic_data


@pytest.mark.parametrize("impurity", Impurity)
def test_sensitivity_to_electron_density_temp(impurity):
    """
    Current setup requires a guess for the electron density + temp to determine
    the charge state to determine the electron density + temp creating a bit of 
    a chicken and egg problem. This test checks that the output of charge state
    calculation is not too sensitive to the guess for the electron density.
    """
    RTOL = 1e-6 
    RTOL_TUNGSTEN_DENS = 2e-4 # Tungsten is more sensitive to the electron density guess.
    RTOL_TUNGSTEN_TEMP = 0.8 # TODO(allenw): Tungsten can be quite sensitive to the electron temperature guess.
    DENSITY_PERTUBATION_FACTORS = jnp.linspace(0.5, 1.5, 10)
    TEMPERATURE_PERTURBATION_FACTORS = jnp.linspace(0.25, 1.25, 10)
    electron_density_19_tests = jnp.linspace(1.0, 40.0, 10)
    electron_temp_kev_tests = jnp.linspace(1.0, 20.0, 10)
    impurity_concentration_tests = jnp.logspace(-4, -1, 10)

    atomic_data = read_atomic_data()
    impurity_calc = ImpurityCalculator(
                impurity,
                atomic_data[impurity].coronal_mean_Z_interpolator,
                atomic_data[impurity].coronal_Lz_interpolator,
            )
            
    @jax.jit
    def calc_maximum_normalized_diff(x):
        """
        Find the maximum difference between the elements of "x".
        Normalize against the mean of "x".
        """
        return (jnp.max(x) - jnp.min(x))/jnp.mean(x)

    """ 
    Test the sensitivity of the charge state to the electron density guess.
    """
    def test_perturb_density(nominal_density_19, electron_temp_kev, impurity_concentration):
        out = impurity_calc(average_electron_density_19=DENSITY_PERTUBATION_FACTORS * nominal_density_19, average_electron_temp_keV=electron_temp_kev, impurity_concentration=impurity_concentration)
        max_norm_diff = jax.tree_util.tree_map(calc_maximum_normalized_diff, out)
        return max_norm_diff
    
    # Essentially a triple for loop over nominal_density_19_tests, electron_temp_kev_tests, and impurity_concentration_tests.
    combinatorial_test_case = jax.vmap(jax.vmap(jax.vmap(test_perturb_density, in_axes=(0, None, None)), in_axes=(None, 0, None)), in_axes=(None, None, 0))
    test_results = combinatorial_test_case(electron_density_19_tests, electron_temp_kev_tests, impurity_concentration_tests)
    for k, v in test_results.items():
        assert jnp.all(v < RTOL_TUNGSTEN_DENS if impurity == Impurity.Tungsten else RTOL)

    """
    Test the sensitivity of the charge state to the electron temperature guess.
    """
    def test_perturb_temperature(nominal_density_19, electron_temp_kev, impurity_concentration):
        out = impurity_calc(average_electron_density_19=nominal_density_19, average_electron_temp_keV=TEMPERATURE_PERTURBATION_FACTORS * electron_temp_kev, impurity_concentration=impurity_concentration)
        max_norm_diff = jax.tree_util.tree_map(calc_maximum_normalized_diff, out)
        return max_norm_diff
    
    # Again, a triple for loop over nominal_density_19_tests, electron_temp_kev_tests, and impurity_concentration_tests.
    combinatorial_test_case = jax.vmap(jax.vmap(jax.vmap(test_perturb_temperature, in_axes=(0, None, None)), in_axes=(None, 0, None)), in_axes=(None, None, 0))
    test_results = combinatorial_test_case(electron_density_19_tests, electron_temp_kev_tests, impurity_concentration_tests)
    for k, v in test_results.items():
        assert jnp.all(v < RTOL_TUNGSTEN_TEMP if impurity == Impurity.Tungsten else RTOL)
    

def test_zeff_and_dilution_from_impurities(load_sparc_prd_data):
    input_parameters, algorithm, points, impurity_types, impurity_concentrations = load_sparc_prd_data


    calc = CalcZeffAndDilutionFromImpurities(impurity_types)


    average_electron_density = 1e19 * input_parameters["average_electron_density"].values
    average_electron_temp = 1000.0 * input_parameters["average_electron_temp"].values

    # Create a grid of inputs.
    average_electron_density_grid, average_electron_temp_grid = jnp.meshgrid(average_electron_density, average_electron_temp)


    # Double vectorize to allow for input of 2D arrays.
    # The first vmap is over rows of the 2D array and feeds a 1D array to the second vmap.
    vectorized_calc = jax.vmap(jax.vmap(calc, in_axes=(0, 0, None)), in_axes=(0, 0, None))

    out, debug = vectorized_calc(average_electron_density_grid, average_electron_temp_grid, impurity_concentrations)
    for v in out.values():
        assert v.shape == average_electron_density_grid.shape
        assert v.shape == average_electron_temp_grid.shape

    """Fake profiles for testing calc_impurity_radiated_power_radas."""
    rho = jnp.linspace(0.0, 1.0, 100)
    electron_temp_profile = jnp.linspace(20.0, 2.0, 100)
    electron_density_profile = jnp.linspace(40.0, 10.0, 100)
    plasma_volume = 20.0
    out, debug = calc.calc_impurity_radiated_power_radas(
        rho,
        electron_temp_profile=1e3*electron_temp_profile,
        electron_density_profile=1e19*electron_density_profile,
        impurity_concentrations=impurity_concentrations,
        plasma_volume=plasma_volume,
    )

    # TODO(allenw): seems a bit low?
    assert out > 1.5

    """
    Some numbers approximately correct for the SPARC PRD.
    """
    beta = 0.012
    Bt = 12.2
    # Use equation 4.4 from Creely 2020.
    # This pressure is in keV * 1e19 m^-3.
    average_pressure = (beta * Bt**2.0) / 4.02e-3
    average_ion_density = 27.0 # 1e19 m^-3
    td_calc = TempDensImpurities(zeff_and_dilution_calc=calc)
    out = td_calc(average_pressure=average_pressure,
                          average_ion_density=average_ion_density,
                          ion_to_electron_temp_ratio=1.0,
                          impurity_concentrations=impurity_concentrations)

    # Check that outputs are reasonably close to Creely 2020.
    assert out["average_electron_density"] > 30 and out["average_electron_density"] < 35
    assert out["average_electron_temp"] > 6.5 and out["average_electron_temp"] < 8.0
    assert out["average_ion_density"] > 25 and out["average_ion_density"] < 30
    assert out["average_ion_temp"] > 6.5 and out["average_ion_temp"] < 8.0
    assert out["z_effective"] > 1.0 and out["z_effective"] < 2.0
    assert out["dilution"] > 0.7 and out["dilution"] < 0.9

