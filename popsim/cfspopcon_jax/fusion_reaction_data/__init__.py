"""Reactions rates and power densities for various fusion reactions."""
from enum import IntEnum
from typing import Callable, Union

from numpy import float64
from numpy.typing import NDArray

from .reaction_energies import reaction_energy_DD, reaction_energy_DHe3, reaction_energy_DT
from .reaction_rate_coefficients import sigmav_DD, sigmav_DD_BoschHale, sigmav_DHe3, sigmav_DT, sigmav_DT_BoschHale

SIGMAV_FUNC = Union[
    Callable[[NDArray[float64]], NDArray[float64]],
    Callable[[NDArray[float64]], tuple[NDArray[float64], NDArray[float64], NDArray[float64]]],
]

ENERGY_FUNC = Union[
    Callable[[NDArray[float64], float], tuple[float, float, float, NDArray[float64], NDArray[float64], NDArray[float64]]],
    Callable[
        [tuple[NDArray[float64], NDArray[float64], NDArray[float64]]],
        tuple[NDArray[float64], float, NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]],
    ],
]


class ReactionType(IntEnum):
    """Enum for the various fusion reactions."""

    DT = 0
    DD = 1
    DHe3 = 2


SIGMAV: dict[
    ReactionType,
    SIGMAV_FUNC,
] = {
    ReactionType.DT: sigmav_DT,
    ReactionType.DD: sigmav_DD_BoschHale,
    ReactionType.DHe3: sigmav_DHe3,
}

ENERGY: dict[
    ReactionType,
    ENERGY_FUNC,
] = {
    ReactionType.DT: reaction_energy_DT,
    ReactionType.DD: reaction_energy_DD,
    ReactionType.DHe3: reaction_energy_DHe3,
}

__all__ = [
    "SIGMAV",
    "ENERGY",
    "reaction_energy_DD",
    "reaction_energy_DHe3",
    "reaction_energy_DT",
    "sigmav_DD_BoschHale",
    "sigmav_DHe3",
    "sigmav_DT",
    "sigmav_DD",
    "sigmav_DT_BoschHale",
]
