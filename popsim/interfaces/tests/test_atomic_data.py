import os

import jax
import jax.numpy as jnp
import pytest
import xarray as xr

import cfspopcon
from cfspopcon.formulas import impurity_effects
from popsim import PACKAGE_ROOT
from popsim.interfaces import atomic_data
from popsim.interfaces.enums import Impurity
from popsim.interfaces.zeff_and_dilution_from_impurities import (
    CalcZeffAndDilutionFromImpurities,
)


@pytest.fixture
def load_data():
    sparc_prd_path = os.path.join(PACKAGE_ROOT, "../cfspopcon/example_cases/SPARC_PRD")
    input_parameters, algorithm, points = cfspopcon.read_case(sparc_prd_path)
    algorithm.validate_inputs(input_parameters)
    return input_parameters, algorithm, points


def test_zeff_and_dilution_from_impurities(load_data):
    input_parameters, algorithm, points = load_data

    impurity_types = [Impurity(impurity.value) for impurity in input_parameters["impurities"].dim_species.data]

    calc = CalcZeffAndDilutionFromImpurities(impurity_types)

    impurity_concentrations = dict(
        zip(
            impurity_types,
            input_parameters["impurities"].values,
        )
    )

    average_electron_density = 1e19 * input_parameters["average_electron_density"].values
    average_electron_temp = 1000.0 * input_parameters["average_electron_temp"].values

    # Create a grid of inputs.
    average_electron_density_grid, average_electron_temp_grid = jnp.meshgrid(average_electron_density, average_electron_temp)


    # Double vectorize to allow for input of 2D arrays.
    # The first vmap is over rows of the 2D array and feeds a 1D array to the second vmap.
    vectorized_calc = jax.vmap(jax.vmap(calc, in_axes=(0, 0, None)), in_axes=(0, 0, None))

    out = vectorized_calc(average_electron_density_grid, average_electron_temp_grid, impurity_concentrations)
    for v in out.values():
        assert v.shape == average_electron_density_grid.shape
        assert v.shape == average_electron_temp_grid.shape



def test_interpolator_modes(load_data):
    input_parameters, algorithm, points = load_data

    cfspopcon.algorithms.calc_zeff_and_dilution_from_impurities.validate_inputs(input_parameters)
    dataset = xr.Dataset(input_parameters)
    cfspopcon.algorithms.calc_zeff_and_dilution_from_impurities.update_dataset(dataset, in_place=True)

    atomic_data_cfspopcon = cfspopcon.atomic_data.read_atomic_data()
    atomic_data_popsim = atomic_data.read_atomic_data()

    # Need the .value because the former uses enum and the latter uses IntEnum.
    assert [k.value for k in atomic_data_cfspopcon.keys()] == [k.value for k in atomic_data_popsim.keys()]

    test_temp = 1e4 # eV
    test_density = 1e20 # m^-3
    for species_cfs, species_popsim in zip(atomic_data_cfspopcon.keys(), atomic_data_popsim.keys()):

        # Test that the interpolators are the same.
        cfspopcon_charge_state = impurity_effects.calc_impurity_charge_state_impl(
            test_density, test_temp, atomic_data_cfspopcon[species_cfs].coronal_Lz_interpolator
        )

        popsim_charge_state = impurity_effects.calc_impurity_charge_state_impl(
            test_density, test_temp, atomic_data_popsim[species_popsim].coronal_Lz_interpolator
        )

        assert jnp.isclose(cfspopcon_charge_state, popsim_charge_state)
