import jax
import jax.numpy as jnp

from popsim.algorithms.zeff_and_dilution_from_impurities import CalcZeffAndDilutionFromImpurities, CalcTempDensBreakdown
from popsim.enums import Impurity
from popsim.tests import load_sparc_prd_data


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

    """
    Some numbers approximately correct for the SPARC PRD.
    """
    beta = 0.012
    Bt = 12.2
    # Use equation 4.4 from Creely 2020.
    # This pressure is in keV * 1e19 m^-3.
    average_pressure = (beta * Bt**2.0) / 4.02e-3
    average_ion_density = 27.0 # 1e19 m^-3
    td_calc = CalcTempDensBreakdown(zeff_and_dilution_calc=calc)
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
    import pdb; pdb.set_trace()