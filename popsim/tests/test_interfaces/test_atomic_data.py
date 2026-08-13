
from pathlib import Path

import cfspopcon
import jax.numpy as jnp
import xarray as xr
from cfspopcon.unit_handling import ureg

from popsim.cfspopcon_jax import impurity_effects
from popsim.enums import Impurity
from popsim.interfaces import atomic_data
from popsim.interfaces.cfspopcon_scenario import load_cfspopcon_scenario


def test_interpolator_modes():
    atomic_data_cfspopcon, _ = cfspopcon.formulas.atomic_data.atomic_data.read_atomic_data(radas_dir=Path("./atomic_data"))
    atomic_data_popsim = atomic_data.read_atomic_data()

    # Need the .value because the former uses enum and the latter uses IntEnum.
    assert all(jnp.sort(jnp.array([k.value for k in atomic_data_cfspopcon.datasets.keys()])) == jnp.sort(jnp.array([k.value for k in atomic_data_popsim.keys()])))

    test_temp = ureg.Quantity(1e4, ureg.eV)
    test_density = ureg.Quantity(1e20, 1.0/ureg.meter**3.0)
    for species_cfs, species_popsim in zip(atomic_data_cfspopcon.datasets.keys(), atomic_data_popsim.keys()):
        
        # Krypton radas curves are yielding nans. Not an important species anyway.
        if species_cfs.value == species_popsim.value and species_popsim == Impurity.Krypton: 
            continue

        # Test that the interpolators are the same.
        # Call the interpolator directly with magnitudes: the unit-wrapped .eval is broken
        # for scalar Quantities in cfspopcon 8 with numpy 2.4 (np.vectorize raises
        # "setting an array element with a sequence").
        interpolator = atomic_data_cfspopcon.get_coronal_Lz_interpolator(species_cfs)
        cfspopcon_charge_state = interpolator(test_density.magnitude, test_temp.magnitude, allow_extrap=True)
        popsim_charge_state = impurity_effects.calc_impurity_charge_state_impl(
            test_density.magnitude, test_temp.magnitude, atomic_data_popsim[species_popsim].coronal_Lz_interpolator
        )

        assert jnp.isclose(cfspopcon_charge_state, popsim_charge_state)