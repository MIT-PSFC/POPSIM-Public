import itertools
import typing

import chex
import diffrax
from jaxtyping import Scalar

# Dictionary of times to values.
TrajectoryInput = dict[Scalar, Scalar]
TrajectoryOrTrajectoryInput = diffrax.AbstractPath | TrajectoryInput
ConstantOrTimeDependent = Scalar | TrajectoryOrTrajectoryInput


class CombinatorialCases:
    def __init__(self, config: typing.Sequence[ConstantOrTimeDependent]):
        self.config = config


def generate_combinations(params: chex.dataclass) -> list[chex.dataclass]:
    """Given a dataclass with fields that are instances of CombinatorialCases, generate all combinations of the list fields.

    Args:
        params (chex.dataclass): A dataclass instance with fields that are instances of CombinatorialCases.

    Returns:
        list[chex.dataclass]: A list of dataclass instances with all combinations of the list fields.
    """
    # Identify list fields and their respective values
    list_fields = {
        field: getattr(params, field).config for field in params.__annotations__ if isinstance(getattr(params, field), CombinatorialCases)
    }
    if not list_fields:
        return [params]

    # Generate all combinations of list fields
    keys, values = zip(*list_fields.items())
    combinations = itertools.product(*values)

    # Create a list to hold all combinations of Params instances
    params_combinations = []

    # Get the constructor of the class of 'params'
    constructor = type(params)

    for combination in combinations:
        # Create a dictionary for the current combination
        combo_dict = dict(zip(keys, combination))

        # Create a Params instance for the current combination
        # Update the non-list fields with their original values
        new_params = {field: getattr(params, field) if field not in combo_dict else combo_dict[field] for field in params.__annotations__}
        params_combinations.append(constructor(**new_params))

    return params_combinations


def build_config(config):
    pass


def build_combinatorial_config():
    pass
