import dataclasses
import itertools
import typing

import chex
import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, PyTree

from popsim.tree_util import get_instances_from_tree_leaves


@chex.dataclass
class SimInput:
    time: Array
    initial_state: PyTree
    params: PyTree

    def generate_sim_cases(self) -> list["SimInput"]:
        """See `generate_cases` for more information.

        Returns:
            list["SimInput"]: List of generated SimInput instances.
        """
        return generate_cases(self)


@chex.dataclass
class CombinatorialCases:
    cases: list = dataclasses.field(default_factory=list)


@chex.dataclass
class MultiCases:
    cases: list = dataclasses.field(default_factory=list)


def make_time_base(t0: float, t1: float, dt: float) -> np.ndarray:
    """Create a time base from t0 to t1 with a step size of dt.
    Use numpy as jnp may give unhashable types.

    Args:
        t0 (float): time to start.
        t1 (float): time to end.
        dt (float): time step.

    Returns:
        np.ndarray: time base.
    """
    return np.arange(t0, t1 + dt, dt)


def generate_cases(
    tree: PyTree[typing.Union[typing.Any, MultiCases, CombinatorialCases]],
) -> typing.Union[list[PyTree[typing.Any]], PyTree[typing.Any]]:
    """Given an input tree that may contain nodes that are MultiCases or CombinatorialCases, generate all possible cases.

    Args:
        tree (PyTree[typing.Union[typing.Any, MultiCases]]): A PyTree that may contain nodes that are MultiCases or CombinatorialCases.

    Raises:
        ValueError: error if tree contains both instances of CombinatorialCases and MultiCases.
        ValueError: error if all instances of MultiCases do not have the same length.

    Returns:
        typing.Union[list[PyTree[typing.Any]], PyTree[typing.Any]]: A list of tree instances where all instances of CombinatorialCases or MultiCases have been replaced with their respective values. If no instances of CombinatorialCases or MultiCases are found, the original tree is returned.
    """
    combinatorial_cases = get_instances_from_tree_leaves(tree, CombinatorialCases)
    multi_cases = get_instances_from_tree_leaves(tree, MultiCases)

    if combinatorial_cases and multi_cases:
        raise ValueError(f"{type(tree)} can only contain instances of CombinatorialCases or MultiCases and not both.")

    if multi_cases:
        # All cases must have the same length
        lengths = [len(x.cases) for x in multi_cases]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All instances of MultiCases must have the same length.")
        return generate_multi_cases(tree)
    elif combinatorial_cases:
        return generate_combinatorial_cases(tree)
    else:
        return tree


def generate_multi_cases(tree: PyTree[typing.Union[typing.Any, MultiCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of MultiCases, generate all possible cases. Note that all instances of MultiCases must have the same length.

    Args:
        tree (PyTree[typing.Union[typing.Any, MultiCases]]): PyTree where some leaves are instances of MultiCases.

    Raises:
        ValueError: error if all instances of MultiCases do not have the same length.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of MultiCases have been replaced with their respective values.
    """
    multi_cases = get_instances_from_tree_leaves(tree, MultiCases)

    if not multi_cases:
        return tree

    def is_multi_case(x):
        return isinstance(x, MultiCases)

    # Check that all instances of MultiCases have the same length
    lengths = [len(x.cases) for x in multi_cases]

    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All instances of MultiCases must have the same length.")

    # Partition the tree into MultiCases and non-MultiCases
    multi_cases_tree, non_multi_cases_tree = eqx.partition(tree, is_multi_case, is_leaf=is_multi_case)

    list_of_multi_cases, treedef = jax.tree.flatten(multi_cases_tree, is_leaf=is_multi_case)

    list_of_multi_cases_list = [x.cases for x in list_of_multi_cases]

    cases = list(zip(*list_of_multi_cases_list))

    def reconstruct_tree(case):
        reconstructed_multi_case_tree = jax.tree.unflatten(treedef, case)
        return eqx.combine(non_multi_cases_tree, reconstructed_multi_case_tree)

    reconstructed_trees = [reconstruct_tree(case) for case in cases]
    return reconstructed_trees


def generate_combinatorial_cases(tree: PyTree[typing.Union[typing.Any, CombinatorialCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of CombinatorialCases, generate all combinations of the fields of the CombinatorialCases.

    Args:
        tree (PyTree[typing.Union[typing.Any, CombinatorialCases]]): PyTree where some leaves are instances of CombinatorialCases.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of CombinatorialCases have been replaced with various combinations of their fields.
    """
    comb_cases = get_instances_from_tree_leaves(tree, CombinatorialCases)
    if not comb_cases:
        return tree

    def is_comb_case(x):
        return isinstance(x, CombinatorialCases)

    # Partition the tree into CombinatorialCases and non-CombinatorialCases
    comb_cases_tree, non_comb_cases_tree = eqx.partition(tree, is_comb_case, is_leaf=is_comb_case)

    list_of_comb_cases, treedef = jax.tree.flatten(comb_cases_tree, is_leaf=is_comb_case)

    list_of_comb_cases_list = [x.cases for x in list_of_comb_cases]

    combinations = list(itertools.product(*list_of_comb_cases_list))

    def reconstruct_tree(comb):
        reconstructed_comb_cases_tree = jax.tree.unflatten(treedef, comb)
        return eqx.combine(non_comb_cases_tree, reconstructed_comb_cases_tree)

    reconstructed_trees = [reconstruct_tree(comb) for comb in combinations]
    return reconstructed_trees
