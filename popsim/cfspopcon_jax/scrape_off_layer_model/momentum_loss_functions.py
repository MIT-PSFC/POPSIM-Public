"""Calculate SOL momentum loss as a function target electron temperature.

See Figure 15 of :cite:`stangeby_2018`.
"""
import jax.numpy as np
from cfspopcon.named_options import MomentumLossFunction
from cfspopcon.unit_handling import Quantity
from numpy import float64
from numpy.typing import NDArray


def _calc_SOL_momentum_loss_fraction(A: float, Tstar: float, n: float, target_electron_temp: float) -> float:
    """Calculates the fraction of momentum lost in the SOL, for a generic SOL momentum loss function.

    This is equation 33 of :cite:`stangeby_2018`, rearranged for $f^{total}_{mom-loss}$
    """
    return 1.0 - A * (1.0 - np.exp(-target_electron_temp / Tstar)) ** n


def calc_SOL_momentum_loss_fraction(key: MomentumLossFunction, target_electron_temp: Quantity) -> float:
    """Calculate the fraction of momentum lost in the SOL.

    The coefficients come from figure captions in :cite:`stangeby_2018`
    * KotovReiter: SOLPS scans with Deuterium only, no impurities for JET vertical target. Figure 7a)
    * Sang: SOLPS scans for Deuterium with Carbon impurity, for DIII-D, variety of divertor configurations. Figure 7b)
    * Jarvinen: EDGE2D density scan for JET with a horizontal target, for a variety of targets and injected impurities. Figure 10a)
    * Moulton: SOLPS density scan for narrow slot divertor, no impurities. Figure 10b)
    * PerezH: SOLPS density scan for AUG H-mode, only trace impurities. Figure 11a)
    * PerezL: SOLPS density scan for AUG L-mode, with Carbon impurity. Figure 11b)

    Comparison is in Figure 15.

    Args:
        key: which momentum loss function to use
        target_electron_temp: electron temperature at the target [eV]

    Returns:
        SOL_momentum_loss_fraction [~]
    """
    if key == MomentumLossFunction.KotovReiter:
        return _calc_SOL_momentum_loss_fraction(1.0, 0.8, 2.1, target_electron_temp)

    elif key == MomentumLossFunction.Sang:
        return _calc_SOL_momentum_loss_fraction(1.3, 1.8, 1.6, target_electron_temp)

    elif key == MomentumLossFunction.Jarvinen:
        return _calc_SOL_momentum_loss_fraction(1.7, 2.2, 1.2, target_electron_temp)

    elif key == MomentumLossFunction.Moulton:
        return _calc_SOL_momentum_loss_fraction(1.0, 1.0, 1.5, target_electron_temp)

    elif key == MomentumLossFunction.PerezH:
        return _calc_SOL_momentum_loss_fraction(0.8, 2.0, 1.2, target_electron_temp)

    elif key == MomentumLossFunction.PerezL:
        return _calc_SOL_momentum_loss_fraction(1.1, 3.0, 0.9, target_electron_temp)

    else:
        raise NotImplementedError(f"No implementation for MomentumLossFunction {key}")


def get_SOL_momentum_loss_function(key: MomentumLossFunction):
    inputs = {
        MomentumLossFunction.KotovReiter: np.array([1.0, 0.8, 2.1]),
        MomentumLossFunction.Sang: np.array([1.3, 1.8, 1.6]),
        MomentumLossFunction.Jarvinen: np.array([1.7, 2.2, 1.2]),
        MomentumLossFunction.Moulton: np.array([1.0, 1.0, 1.5]),
        MomentumLossFunction.PerezH: np.array([0.8, 2.0, 1.2]),
        MomentumLossFunction.PerezL: np.array([1.1, 3.0, 0.9]),
    }

    return inputs[key]


def calc_SOL_momentum_loss_function_array(inputs: NDArray[float64], target_electron_temp):
    return _calc_SOL_momentum_loss_fraction(inputs[0], inputs[1], inputs[2], target_electron_temp)
