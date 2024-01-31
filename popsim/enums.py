"""Jax-compatible enums."""
from enum import IntEnum
from typing import Union

import cfspopcon.named_options as cfsno


def enum_to_intenum(enum_class):
    """Converts a given Enum class to an IntEnum class.

    The values of the original Enum are preserved.
    """
    # Create a dictionary of the enum members and their values
    members = {name: value.value for name, value in enum_class.__members__.items()}

    # Create a new IntEnum class with the same name and members
    return IntEnum(enum_class.__name__, members)


class FuelSpecies(IntEnum):
    """
    Fuel species where the integer value corresponds to the atomic mass number.
    """

    Deuterium = 2
    Tritium = 3


Impurity = enum_to_intenum(cfsno.Impurity)
Species = Union[FuelSpecies, Impurity]
ProfileForm = enum_to_intenum(cfsno.ProfileForm)


# Map from Species to atomic number.
AtomicNumberMap = {value: value.value for _, value in Impurity.__members__.items()} | {
    FuelSpecies.Deuterium: 1,
    FuelSpecies.Tritium: 1,
}
