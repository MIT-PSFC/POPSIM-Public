import dataclasses
import itertools
import typing

import chex
import equinox as eqx
import jax
from jaxtyping import PyTree

import popsim.interp as pinterp
import popsim.types as ptypes


@chex.dataclass
class CombinatorialCases:
    config: list = dataclasses.field(default_factory=list)


@chex.dataclass
class MultiCases:
    config: list = dataclasses.field(default_factory=list)


def get_cases(tree, type_):
    def func(x):
        return isinstance(x, type_)

    return [x for x in jax.tree.leaves(tree, func) if func(x)]


def generate_multi_cases(config: PyTree[typing.Union[typing.Any, MultiCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of MultiCases, generate all possible cases.
    Note that all instances of MultiCases must have the same length.

    Args:
        config (PyTree[typing.Union[typing.Any, MultiCases]]): PyTree where some leaves are instances of MultiCases.

    Raises:
        ValueError: error if all instances of MultiCases do not have the same length.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of MultiCases have been replaced with their respective values.
    """
    multi_cases = get_cases(config, MultiCases)

    if not multi_cases:
        return config

    def is_multi_case(x):
        return isinstance(x, MultiCases)

    # Check that all instances of MultiCases have the same length
    lengths = [len(x.config) for x in multi_cases]

    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All instances of MultiCases must have the same length.")

    # Partition the tree into MultiCases and non-MultiCases
    multi_cases_tree, non_multi_cases_tree = eqx.partition(config, is_multi_case, is_leaf=is_multi_case)

    list_of_multi_cases, treedef = jax.tree.flatten(multi_cases_tree, is_leaf=is_multi_case)

    list_of_multi_cases_list = [x.config for x in list_of_multi_cases]

    cases = list(zip(*list_of_multi_cases_list))

    def reconstruct_tree(case):
        reconstructed_multi_case_tree = jax.tree.unflatten(treedef, case)
        return eqx.combine(non_multi_cases_tree, reconstructed_multi_case_tree)

    reconstructed_trees = [reconstruct_tree(case) for case in cases]
    return reconstructed_trees


def generate_combinatorial_cases(config: PyTree[typing.Union[typing.Any, CombinatorialCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of CombinatorialCases, generate all combinations of the fields of the CombinatorialCases.

    Args:
        config (PyTree[typing.Union[typing.Any, CombinatorialCases]]): PyTree where some leaves are instances of CombinatorialCases.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of CombinatorialCases have been replaced with various combinations of their fields.
    """
    comb_cases = get_cases(config, CombinatorialCases)
    if not comb_cases:
        return config

    def is_comb_case(x):
        return isinstance(x, CombinatorialCases)

    # Partition the tree into CombinatorialCases and non-CombinatorialCases
    comb_cases_tree, non_comb_cases_tree = eqx.partition(config, is_comb_case, is_leaf=is_comb_case)

    list_of_comb_cases, treedef = jax.tree.flatten(comb_cases_tree, is_leaf=is_comb_case)

    list_of_comb_cases_list = [x.config for x in list_of_comb_cases]

    combinations = list(itertools.product(*list_of_comb_cases_list))

    def reconstruct_tree(comb):
        reconstructed_comb_cases_tree = jax.tree.unflatten(treedef, comb)
        return eqx.combine(non_comb_cases_tree, reconstructed_comb_cases_tree)

    reconstructed_trees = [reconstruct_tree(comb) for comb in combinations]
    return reconstructed_trees


def build_config_paths(
    config: PyTree[ptypes.ConstantOrTimeDependentSpec], interp_type: str = "linear"
) -> PyTree[ptypes.ConstantOrTimeDependent]:
    """Given a user-specified configuration of the form "ConstantOrTimeDependentSpec", turn instances of "TrajectorySpec" into
    instances of "diffrax.AbstractPath" by interpolating the values at the specified times.

    Args:
        config (PyTree[ptypes.ConstantOrTimeDependentSpec]): A user-specified configuration.
        interp_type (str): The interpolation type. Can be "linear" or "cubic". Defaults to "linear".

    Returns:
        PyTree[ptypes.ConstantOrTimeDependent]: A configuration where all instances of "TrajectorySpec" have been interpolated.
    """

    def is_traj_spec(x):
        # I would prefer to do a:
        #   isinstance(x, ptypes.TrajectorySpec)  # noqa: ERA001
        # but isinstance does not work with paramerized generics.
        # Instead, we will check that "x" is a dictionary with float keys.
        # We will also check that all values have the same tree structure.
        # First, check if 'x' is a dictionary.
        if not isinstance(x, dict):
            return False

        # Check if all keys are floats
        keys_all_floats = all(isinstance(key, float) for key in x.keys())
        if not keys_all_floats:
            return False

        # Check if all values have the same tree structure
        values = list(x.values())
        structures = [jax.tree.structure(v) for v in values]
        return all(structure == structures[0] for structure in structures)

    def maybe_interp(x):
        return x if not is_traj_spec(x) else pinterp.interp_time_dic(x, interp_type=interp_type)

    return jax.tree_map(maybe_interp, config, is_leaf=is_traj_spec)


def check_config(config: PyTree[ptypes.ConstantOrTimeDependentSpec]) -> None:
    """Check the config for validity.
    The rules are:
        1) config can only contain instances of CombinatorialCases or MultiCases and not both.

    Args:
        config (PyTree[ptypes.ConstantOrTimeDependentSpec]): The configuration to check.
    """

    combinatorial_cases = get_cases(config, CombinatorialCases)
    multi_cases = get_cases(config, MultiCases)

    if combinatorial_cases and multi_cases:
        raise ValueError("config can only contain instances of CombinatorialCases or MultiCases and not both.")
    if multi_cases:
        # All cases must have the same length
        lengths = [len(x.config) for x in multi_cases]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All instances of MultiCases must have the same length.")

    out = {
        "combinatorial_cases": combinatorial_cases,
        "multi_cases": multi_cases,
    }
    return out


def build_configs(
    config: PyTree[ptypes.ConstantOrTimeDependentSpec], interp_type: str = "linear"
) -> typing.Union[list[PyTree[ptypes.ConstantOrTimeDependent]], PyTree[ptypes.ConstantOrTimeDependent]]:
    """Given a config PyTree, generate simulation-ready configurations. This involves two steps:
        1) Interpolating all instances of TrajectorySpec for all CombinatorialCases and MultiCases.
        2) Resolving all instances of CombinatorialCases and MultiCases.

    Args:
        config (PyTree[ptypes.ConstantOrTimeDependentSpec]): _description_

    Returns:
        list[PyTree[ptypes.ConstantOrTimeDependent]]: _description_
    """
    out = check_config(config)
    interped = build_config_paths(config, interp_type)

    if out["combinatorial_cases"]:
        # Generate all combinations of CombinatorialCases
        combinations = generate_combinatorial_cases(interped)
        return combinations
    elif out["multi_cases"]:
        pass
    else:
        return interped
