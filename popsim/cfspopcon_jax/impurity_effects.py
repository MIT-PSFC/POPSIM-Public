from typing import Callable

import xarray as xr
from cfspopcon.np_variant import np


def calc_impurity_charge_state_impl(
    average_electron_density: float, average_electron_temp: float, mean_charge_curve: Callable[[float], float]
) -> float:
    """Calculate the impurity charge state of the specified impurity species.

    Args:
        average_electron_density: [m^-3] :term:`glossary link<average_electron_density>`
        average_electron_temp: [eV] :term:`glossary link<average_electron_temp>`
        mean_charge_curve (Callable[[float], float]): _description_

    Returns:
        float: the impurity charge state.
    """
    return np.squeeze(np.power(10, mean_charge_curve(np.log10(average_electron_temp), np.log10(average_electron_density))))


def calc_change_in_zeff(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in the effective charge due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change in zeff [~]
    """
    return impurity_charge_state * (impurity_charge_state - 1.0) * impurity_concentration


def calc_change_in_dilution(impurity_charge_state: float, impurity_concentration: xr.DataArray) -> xr.DataArray:
    """Calculate the change in n_fuel/n_e due to the specified impurities.

    Args:
        impurity_charge_state: [~] :term:`glossary link<impurity_charge_state>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`

    Returns:
        change in dilution [~]
    """
    return impurity_charge_state * impurity_concentration
