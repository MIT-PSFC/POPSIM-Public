import jax
import jax.numpy as jnp

from popsim.algorithms.zeff_and_dilution_from_impurities import CalcZeffAndDilutionFromImpurities
from popsim.enums import Impurity


def test_zeff_and_dilution_from_impurities(load_sparc_prd_data):
    input_parameters, algorithm, points = load_sparc_prd_data

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
