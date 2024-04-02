import jax
import jax.numpy as jnp
import pytest

from popsim.algorithms.impurities import calc_impurity_charge_state
from popsim.enums import Impurity
from popsim.interfaces.atomic_data import read_atomic_data

radas_curves = read_atomic_data()

@pytest.mark.parametrize("impurity", Impurity)
def test_sensitivity_to_electron_density_temp(impurity):
    """
    Current setup requires a guess for the electron density + temp to determine
    the charge state to determine the electron density + temp creating a bit of
    a chicken and egg problem. This test checks that the output of charge state
    calculation is not too sensitive to the guess for the electron density + temperature.
    """
    RTOL = 1e-6
    RTOL_TUNGSTEN_DENS = 2e-4 # Tungsten is more sensitive to the electron density guess.
    RTOL_TUNGSTEN_TEMP = 0.8 # TODO(allenw): Tungsten can be quite sensitive to the electron temperature guess.
    DENSITY_PERTUBATION_FACTORS = jnp.linspace(0.5, 1.5, 10)
    TEMPERATURE_PERTURBATION_FACTORS = jnp.linspace(0.5, 1.5, 10)
    electron_density_19_tests = jnp.linspace(1.0, 40.0, 10)
    electron_temp_kev_tests = jnp.linspace(1.0, 20.0, 10)


    @jax.jit
    def calc_maximum_normalized_diff(x):
        """
        Find the maximum difference between the elements of "x".
        Normalize against the mean of "x".
        """
        return (jnp.max(x) - jnp.min(x))/jnp.mean(x)

    """
    Test the sensitivity of the charge state to the electron density guess at a range of electron densities + temperatures.
    """
    def test_perturb_density(nominal_density_19, electron_temp_kev):
        charge_states = calc_impurity_charge_state(average_electron_density_19=DENSITY_PERTUBATION_FACTORS * nominal_density_19, average_electron_temp_keV=electron_temp_kev, radas_curves=radas_curves[impurity])
        out = calc_maximum_normalized_diff(charge_states)
        return out

    # Essentially a triple for loop over nominal_density_19_tests, electron_temp_kev_tests.
    combinatorial_test_case = jax.vmap(jax.vmap(test_perturb_density, in_axes=(0, None)), in_axes=(None, 0))
    test_results = combinatorial_test_case(electron_density_19_tests, electron_temp_kev_tests)
    assert jnp.all(test_results < RTOL_TUNGSTEN_DENS) if impurity == Impurity.Tungsten else RTOL

    """
    Test the sensitivity of the charge state to the electron temperature guess.
    """
    def test_perturb_temperature(nominal_density_19, electron_temp_kev):
        charge_states = calc_impurity_charge_state(average_electron_density_19=nominal_density_19, average_electron_temp_keV=TEMPERATURE_PERTURBATION_FACTORS * electron_temp_kev, radas_curves=radas_curves[impurity])
        return calc_maximum_normalized_diff(charge_states)

    # Again, a triple for loop over nominal_density_19_tests, electron_temp_kev_tests, and impurity_concentration_tests.
    combinatorial_test_case = jax.vmap(jax.vmap(test_perturb_temperature, in_axes=(0, None)), in_axes=(None, 0))
    test_results = combinatorial_test_case(electron_density_19_tests, electron_temp_kev_tests)
    assert jnp.all(test_results < RTOL_TUNGSTEN_TEMP) if impurity == Impurity.Tungsten else RTOL
