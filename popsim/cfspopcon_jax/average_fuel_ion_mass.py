"""Calculate the average fuel mass in atomic mass units."""

from collections.abc import Callable

from .fusion_rates import ReactionType

FUEL_MASS_AMU: dict[ReactionType, Callable[[float], float]] = {
    ReactionType.DT: lambda heavier_fuel_species_fraction: 2.0 + heavier_fuel_species_fraction,
    ReactionType.DD: lambda _: 2.0,
    ReactionType.DHe3: lambda heavier_fuel_species_fraction: 2.0 + heavier_fuel_species_fraction,
}


def calc_fuel_average_mass_number(fusion_reaction: ReactionType, heavier_fuel_species_fraction: float) -> float:
    """Calculate the average mass of the fuel ions, based on reaction type and fuel mixture ratio.

    Args:
        fusion_reaction: reaction type.
        heavier_fuel_species_fraction: n_heavier / (n_heavier + n_lighter) number fraction.

    Returns:
        :term:`fuel_average_mass_number` [amu]
    """
    return FUEL_MASS_AMU[fusion_reaction](heavier_fuel_species_fraction)
