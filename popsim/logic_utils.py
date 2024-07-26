import typing

import jax.numpy as jnp
from jaxtyping import ArrayLike

"""
Collection of utilities to assist in Jax-compatible logical operations.
"""


def select_w_tuples(conditions_and_choices: typing.Sequence[tuple[bool, ArrayLike]], default: ArrayLike = 0.0) -> ArrayLike:
    """
    Helper function to help you write if elif else statements again. This is really just a thin wrapper around
    jnp.select to choose an array based on a sequence of tuples, where each tuple is a condition and a choice.
    For example, if you want to write an if statement that looks like:

        if condition1:
            return choice1
        elif condition2:
            return choice2
        else:
            return default

    You can instead write:

        conditions_and_choices = [
            (condition1, choice1),
            (condition2, choice2),
        ]
        return select_w_tuples(conditions_and_choices, default=default)


    Args:
        conditions_to_choices (dict[bool, ArrayLike]): dictionary of conditions to choices.
        default (ArrayLike, optional): the value chosen when all conditions evaluate to false. Defaults to 0.0.

    Returns:
        ArrayLike: the chosen array.
    """
    conditions = jnp.array([condition for condition, _ in conditions_and_choices])
    choices = jnp.array([choice for _, choice in conditions_and_choices])
    return jnp.select(conditions, choices, default=default)
