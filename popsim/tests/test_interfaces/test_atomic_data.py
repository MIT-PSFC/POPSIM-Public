
import jax.numpy as jnp
import xarray as xr

import cfspopcon
from cfspopcon.formulas import impurity_effects
from popsim.interfaces import atomic_data
from popsim.scenarios.sparc_prd.generic import load_cfspopcon_prd

def test_interpolator_modes():
    input_parameters, *_= load_cfspopcon_prd()

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
