import jax
import jax.numpy as jnp
from popsim.tests import load_sparc_prd_data
from popsim.algorithms.zeff_and_dilution_from_impurities import CalcZeffAndDilutionFromImpurities, TempDensImpurities


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

