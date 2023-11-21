"""Jax-compatible enums."""
from enum import IntEnum

import cfspopcon.named_options as cfsno


def enum_to_intenum(enum_class):
    """Converts a given Enum class to an IntEnum class.

    The values of the original Enum are preserved.
    """
    # Create a dictionary of the enum members and their values
    members = {name: value.value for name, value in enum_class.__members__.items()}

    # Create a new IntEnum class with the same name and members
    return IntEnum(enum_class.__name__, members)


Impurity = enum_to_intenum(cfsno.Impurity)
