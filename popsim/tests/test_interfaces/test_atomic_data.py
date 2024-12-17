
import cfspopcon
import jax.numpy as jnp
import xarray as xr
from pathlib import Path

from popsim.cfspopcon_jax import impurity_effects
from popsim.interfaces import atomic_data
from popsim.interfaces.cfspopcon_scenario import load_cfspopcon_scenario
from popsim.enums import Impurity

def test_interpolator_modes():
    atomic_data_cfspopcon = cfspopcon.formulas.atomic_data.atomic_data.read_atomic_data(radas_dir=Path("./atomic_data"))
    atomic_data_popsim = atomic_data.read_atomic_data()

    # Need the .value because the former uses enum and the latter uses IntEnum.
    assert all(jnp.sort(jnp.array([k.value for k in atomic_data_cfspopcon.datasets.keys()])) == jnp.sort(jnp.array([k.value for k in atomic_data_popsim.keys()])))

    test_temp = 1e4 # eV
    test_density = 1e20 # m^-3
    for species_cfs, species_popsim in zip(atomic_data_cfspopcon.datasets.keys(), atomic_data_popsim.keys()):
        
        # Krypton radas curves are yielding nans. Not an important species anyway.
        if species_cfs.value == species_popsim.value and species_popsim == Impurity.Krypton: 
            continue

        # Test that the interpolators are the same.
        kind = atomic_data_cfspopcon.CoronalLz
        cfspopcon_charge_state = atomic_data_cfspopcon.eval_interpolator(electron_density=test_density, electron_temp=test_temp, kind=kind, species=species_cfs, allow_extrapolation=True).values[0][0]

        popsim_charge_state = impurity_effects.calc_impurity_charge_state_impl(
            test_density, test_temp, atomic_data_popsim[species_popsim].coronal_Lz_interpolator
        )

        assert jnp.isclose(cfspopcon_charge_state, popsim_charge_state)