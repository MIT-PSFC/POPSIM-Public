"""Calculate the impurity radiated power using the radas atomic_data."""
from typing import Callable

import jax.numpy as np
from numpy import float64
from numpy.typing import NDArray


def calc_impurity_radiated_power_radas(
    electron_temp_profile: NDArray[float64],
    electron_density_profile: NDArray[float64],
    impurity_concentration: float,
    volume_integrator: Callable[[NDArray[float64]], float],
    Lz_curve: Callable[[float64, float64], float64],
) -> float:
    """Calculation of radiated power using radas atomic_data datasets.

    Args:
        rho: [~] :term:`glossary link<rho>`
        electron_temp_profile: [eV] :term:`glossary link<electron_temp_profile>`
        electron_density_profile: [m^-3] :term:`glossary link<electron_density_profile>`
        impurity_species: [] :term:`glossary link<impurity_species>`
        impurity_concentration: [~] :term:`glossary link<impurity_concentration>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`
        Lz_curve: :term:

    Returns:
         [MW] Estimated radiation power due to this impurity
    """
    MW_per_W = 1e6

    Lz = np.power(10, Lz_curve(np.log10(electron_temp_profile), np.log10(electron_density_profile)))
    radiated_power_profile = Lz * electron_density_profile * electron_density_profile

    radiated_power = impurity_concentration * volume_integrator(radiated_power_profile) / MW_per_W
    return radiated_power
