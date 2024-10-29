"""Calculate the thermal gain factor (Q, Q_plasma, Q_thermal)."""
from ..np_variant import np

_IGNITED_THRESHOLD = 1e3
_IGNITED = 1e6


def _ignition_above_threshold(Q: float) -> float:
    """If Q > _IGNITED_THRESHOLD, set equal to _IGNITED.

    Args:
        Q: Fusion power gain [~]

    Returns:
         Q [~]
    """
    return np.where(Q > _IGNITED_THRESHOLD, _IGNITED, Q)


def thermal_calc_gain_factor(P_fusion: float, P_launched: float) -> float:
    """Calculate the fusion gain.

    Args:
        P_fusion: [MW] :term:`glossary link<P_fusion>`
        P_launched: [MW] :term:`glossary link<P_launched>`

    Returns:
         Q [~]
    """
    return np.where(np.isclose(P_launched, 0.0), _IGNITED, _ignition_above_threshold(P_fusion / P_launched))
