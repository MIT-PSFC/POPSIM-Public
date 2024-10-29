"""Common functionality shared between other functions."""

import jax
from numpy import float64
from numpy.typing import NDArray

from ..np_variant import np


def integrate_profile_over_volume_cylindrical(
    array_per_m3: NDArray[float64],
    rho: NDArray[float64],
    plasma_volume: float,
) -> float:
    """Approximate the volume integral of a profile given as a function of rho. Assumes cylindrical geometry.

    Args:
        array_per_m3: a profile of values [units * m^-3]
        rho: [~] :term:`glossary link<rho>`
        plasma_volume: [m^3] :term:`glossary link<plasma_volume>`

    Returns:
         volume_integrated_value [units]
    """
    drho = rho[1] - rho[0]
    return np.sum(array_per_m3 * 2.0 * rho * drho) * plasma_volume


def integrate_profile_over_volume(
    array_per_m3: NDArray[float64],
    rho: NDArray[float64],
    dV_drho: NDArray[float64],
) -> float:
    """Integrate a profile given as a function of rho over the volume. Use the trapezoidal rule.

    Args:
        array_per_m3 (NDArray[float64]): _description_
        rho (NDArray[float64]): _description_
        dV_drho (NDArray[float64]): _description_

    Returns:
        volume_integrated_value [units]
    """
    quantity = np.multiply(array_per_m3, dV_drho)
    return jax.scipy.integrate.trapezoid(quantity, x=rho)
