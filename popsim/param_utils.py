import dataclasses
import itertools
import typing

import chex
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

import popsim.interp as pinterp
import popsim.types as ptypes
from popsim.tree_util import tree_transpose


@chex.dataclass
class CombinatorialCases:
    cases: list = dataclasses.field(default_factory=list)


@chex.dataclass
class MultiCases:
    cases: list = dataclasses.field(default_factory=list)


def get_cases(tree, type_):
    def func(x):
        return isinstance(x, type_)

    return [x for x in jax.tree.leaves(tree, func) if func(x)]


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


def generate_multi_cases(params: PyTree[typing.Union[typing.Any, MultiCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of MultiCases, generate all possible cases.
    Note that all instances of MultiCases must have the same length.

    Args:
        params (PyTree[typing.Union[typing.Any, MultiCases]]): PyTree where some leaves are instances of MultiCases.

    Raises:
        ValueError: error if all instances of MultiCases do not have the same length.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of MultiCases have been replaced with their respective values.
    """
    multi_cases = get_cases(params, MultiCases)

    if not multi_cases:
        return params

    def is_multi_case(x):
        return isinstance(x, MultiCases)

    # Check that all instances of MultiCases have the same length
    lengths = [len(x.cases) for x in multi_cases]

    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All instances of MultiCases must have the same length.")

    # Partition the tree into MultiCases and non-MultiCases
    multi_cases_tree, non_multi_cases_tree = eqx.partition(params, is_multi_case, is_leaf=is_multi_case)

    list_of_multi_cases, treedef = jax.tree.flatten(multi_cases_tree, is_leaf=is_multi_case)

    list_of_multi_cases_list = [x.cases for x in list_of_multi_cases]

    cases = list(zip(*list_of_multi_cases_list))

    def reconstruct_tree(case):
        reconstructed_multi_case_tree = jax.tree.unflatten(treedef, case)
        return eqx.combine(non_multi_cases_tree, reconstructed_multi_case_tree)

    reconstructed_trees = [reconstruct_tree(case) for case in cases]
    return reconstructed_trees


def generate_combinatorial_cases(params: PyTree[typing.Union[typing.Any, CombinatorialCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of CombinatorialCases, generate all combinations of the fields of the CombinatorialCases.

    Args:
        params (PyTree[typing.Union[typing.Any, CombinatorialCases]]): PyTree where some leaves are instances of CombinatorialCases.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of CombinatorialCases have been replaced with various combinations of their fields.
    """
    comb_cases = get_cases(params, CombinatorialCases)
    if not comb_cases:
        return params

    def is_comb_case(x):
        return isinstance(x, CombinatorialCases)

    # Partition the tree into CombinatorialCases and non-CombinatorialCases
    comb_cases_tree, non_comb_cases_tree = eqx.partition(params, is_comb_case, is_leaf=is_comb_case)

    list_of_comb_cases, treedef = jax.tree.flatten(comb_cases_tree, is_leaf=is_comb_case)

    list_of_comb_cases_list = [x.cases for x in list_of_comb_cases]

    combinations = list(itertools.product(*list_of_comb_cases_list))

    def reconstruct_tree(comb):
        reconstructed_comb_cases_tree = jax.tree.unflatten(treedef, comb)
        return eqx.combine(non_comb_cases_tree, reconstructed_comb_cases_tree)

    reconstructed_trees = [reconstruct_tree(comb) for comb in combinations]
    return reconstructed_trees


def build_param_paths(
    params: ptypes.ParamSpec, time_base: Array, interp_type: pinterp.InterpType = pinterp.InterpType.LINEAR
) -> PyTree[diffrax.AbstractPath]:
    """Given a user-specified params specification that is a PyTree of "ConstantOrPathSpec", generate
    a new PyTree of "AbstractPath" on the given time_base. Leaves are handeled as follows:
        1) If the leaf is a dictionary with float keys, it is assumed to be a PathSpec and is interpolated onto "time_base".
        2) If the leaf is an instance of AbstractPath, we re-map to "time_base" and interpolate again.
        3) Otherwise, the leaf is assumed to be a constant, is repeated for all times in "time_base", and interpolated.
    Why interpolate everything? This is done to ensure the inputs to jax.jitted functions are all the same shape across
    calls to prevent re-compiling (which is the main computational cost right now). This way, if the user switches from
    specifying a params as a constant to trajectory, the function will not need to be re-compiled.

    Args:
        params (ptypes.ParamSpec): A user-specified param specification.
        interp_type (pinterp.InterpType): The interpolation type. Defaults to pinterp.InterpType.LINEAR.
        time_base (Array): The time base to interpolate the TrajectorySpecs to.

    Returns:
        PyTree[diffrax.AbstractPath]: A tree where all instances of "PathSpec" have been interpolated.
    """

    def is_path_spec(x):
        # I would prefer to do a:
        #   isinstance(x, ptypes.PathSpec)  # noqa: ERA001
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

    def interp_path_spec(path_spec):
        traj_times = list(path_spec.keys())
        # Check that the times are within the time base
        if not all(time in time_base for time in traj_times):
            raise ValueError(f"All times in specification must be within the time base. Times: {traj_times}, Time base: {time_base}.")
        # Pad the path_spec.
        path_spec[time_base[0]] = path_spec[traj_times[0]]
        path_spec[time_base[-1]] = path_spec[traj_times[-1]]

        # Perform an interpolation to map the PathSpec to the time base.
        intermediate_interp = pinterp.interp_time_dic(path_spec, interp_type=interp_type)
        vals_on_time_base = jax.tree.map(
            lambda x: x.evaluate(time_base), intermediate_interp, is_leaf=lambda x: isinstance(x, diffrax.AbstractPath)
        )

        # Return the final interpolation on the time base.
        final_interp = pinterp.interp(time_base, vals_on_time_base, interp_type=interp_type)
        return final_interp

    def interp_onto_timebase(x):
        if is_path_spec(x):
            return interp_path_spec(x)
        elif isinstance(x, diffrax.AbstractPath):
            vals = x.evaluate(time_base)
            return pinterp.interp(time_base, vals, interp_type=interp_type)
        else:
            vals = jnp.array([x for _ in time_base])
            return pinterp.interp(time_base, vals, interp_type=interp_type)

    return jax.tree.map(interp_onto_timebase, params, is_leaf=lambda x: is_path_spec(x) or isinstance(x, diffrax.AbstractPath))


def check_params(params: ptypes.ParamSpec) -> None:
    """Check the params for validity.
    The rules are:
        1) params can only contain instances of CombinatorialCases or MultiCases and not both.

    Args:
        params (ptypes.ParamSpec): The params spec to check.
    """

    combinatorial_cases = get_cases(params, CombinatorialCases)
    multi_cases = get_cases(params, MultiCases)

    if combinatorial_cases and multi_cases:
        raise ValueError("params can only contain instances of CombinatorialCases or MultiCases and not both.")
    if multi_cases:
        # All cases must have the same length
        lengths = [len(x.cases) for x in multi_cases]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All instances of MultiCases must have the same length.")

    out = {
        "combinatorial_cases": combinatorial_cases,
        "multi_cases": multi_cases,
    }
    return out


def build_params(
    params: ptypes.ParamSpec, time_base: Array, interp_type: pinterp.InterpType = pinterp.InterpType.LINEAR
) -> typing.Union[list[PyTree[diffrax.AbstractPath]], PyTree[diffrax.AbstractPath]]:
    """Given a params specification, generate simulation-ready params trees. This involves two steps:
        1) Interpolating all instances of PathSpec for all CombinatorialCases and MultiCases.
        2) Resolving all instances of CombinatorialCases and MultiCases.

    Args:
        params (ptypes.ParamSpec): param specification.
        time_base (Array): time base to interpolate everything to.

    Returns:
        typing.Union[list[PyTree[diffrax.AbstractPath]], PyTree[diffrax.AbstractPath]]: either
            a list of PyTrees where all instances of CombinatorialCases and MultiCases have been resolved, or
            a single PyTree where paths have been interpolated.
    """
    out = check_params(params)
    interped = build_param_paths(params, time_base, interp_type)

    if out["combinatorial_cases"]:
        # Generate all combinations of CombinatorialCases
        combinations = generate_combinatorial_cases(interped)
        return combinations
    elif out["multi_cases"]:
        # Generate all combinations of MultiCases
        combinations = generate_multi_cases(interped)
        return combinations
    else:
        return interped


def build_vectorized_params(
    params: typing.Union[ptypes.ParamSpec, typing.Sequence[ptypes.ParamSpec]],
    time_base: Array,
    interp_type: pinterp.InterpType = pinterp.InterpType.LINEAR,
    prng_key_seed: typing.Optional[int] = None,
) -> tuple[PyTree[diffrax.AbstractPath], bool]:
    """Given a param specification (or a list of them), build the vectorized params PyTree. s.t. jax.vmap can be used.

    Args:
        params (typing.Union[ptypes.ParamSpec, typing.Sequence[ptypes.ParamSpec]]): param specification or list of param specifications.
        time_base (Array): time base to interpolate everything to.
        interp_type (pinterp.InterpType, optional): The interpolation type. Defaults to pinterp.InterpType.LINEAR.

    Returns:
        tuple[PyTree[diffrax.AbstractPath], bool]: A vectorized params tree and a bool indicating if the tree is multi-simulation.
    """

    def unpack_lists(inp):
        unpacked_list = []
        for item in inp:
            if isinstance(item, list):
                unpacked_list.extend(item)
            else:
                unpacked_list.append(item)
        return unpacked_list

    # First build the parameters.
    params = [build_params(p, time_base, interp_type) for p in params]
    params = unpack_lists(params)

    # We need to perform a tree-transpose to vectorize the parameters.
    params_vectorized = tree_transpose(params)
    return params_vectorized, len(params) > 1
