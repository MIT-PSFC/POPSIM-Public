import itertools
import typing

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

import popsim.interp as pinterp
import popsim.types as ptypes
from popsim.tree_util import get_instances_from_tree_leaves


def generate_sim_cases(sim_input: ptypes.SimInput) -> list[ptypes.SimInput]:
    combinatorial_cases = get_instances_from_tree_leaves(sim_input, ptypes.CombinatorialCases)
    multi_cases = get_instances_from_tree_leaves(sim_input, ptypes.MultiCases)

    if combinatorial_cases and multi_cases:
        raise ValueError("sim_input can only contain instances of ptypes.CombinatorialCases or ptypes.MultiCases and not both.")

    if multi_cases:
        # All cases must have the same length
        lengths = [len(x.cases) for x in multi_cases]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All instances of ptypes.MultiCases must have the same length.")
        return generate_multi_cases(sim_input)
    elif combinatorial_cases:
        return generate_combinatorial_cases(sim_input)


def generate_multi_cases(params: PyTree[typing.Union[typing.Any, ptypes.MultiCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of ptypes.MultiCases, generate all possible cases. Note that all instances of ptypes.MultiCases must have the same length.

    Args:
        params (PyTree[typing.Union[typing.Any, ptypes.MultiCases]]): PyTree where some leaves are instances of ptypes.MultiCases.

    Raises:
        ValueError: error if all instances of ptypes.MultiCases do not have the same length.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of ptypes.MultiCases have been replaced with their respective values.
    """
    multi_cases = get_instances_from_tree_leaves(params, ptypes.MultiCases)

    if not multi_cases:
        return params

    def is_multi_case(x):
        return isinstance(x, ptypes.MultiCases)

    # Check that all instances of ptypes.MultiCases have the same length
    lengths = [len(x.cases) for x in multi_cases]

    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All instances of ptypes.MultiCases must have the same length.")

    # Partition the tree into ptypes.MultiCases and non-ptypes.MultiCases
    multi_cases_tree, non_multi_cases_tree = eqx.partition(params, is_multi_case, is_leaf=is_multi_case)

    list_of_multi_cases, treedef = jax.tree.flatten(multi_cases_tree, is_leaf=is_multi_case)

    list_of_multi_cases_list = [x.cases for x in list_of_multi_cases]

    cases = list(zip(*list_of_multi_cases_list))

    def reconstruct_tree(case):
        reconstructed_multi_case_tree = jax.tree.unflatten(treedef, case)
        return eqx.combine(non_multi_cases_tree, reconstructed_multi_case_tree)

    reconstructed_trees = [reconstruct_tree(case) for case in cases]
    return reconstructed_trees


def generate_combinatorial_cases(params: PyTree[typing.Union[typing.Any, ptypes.CombinatorialCases]]) -> list[PyTree[typing.Any]]:
    """Given a PyTree with instances of ptypes.CombinatorialCases, generate all combinations of the fields of the ptypes.CombinatorialCases.

    Args:
        params (PyTree[typing.Union[typing.Any, ptypes.CombinatorialCases]]): PyTree where some leaves are instances of ptypes.CombinatorialCases.

    Returns:
        list[PyTree[typing.Any]]: A list of PyTrees where all instances of ptypes.CombinatorialCases have been replaced with various combinations of their fields.
    """
    comb_cases = get_instances_from_tree_leaves(params, ptypes.CombinatorialCases)
    if not comb_cases:
        return params

    def is_comb_case(x):
        return isinstance(x, ptypes.CombinatorialCases)

    # Partition the tree into ptypes.CombinatorialCases and non-ptypes.CombinatorialCases
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


def param_specs_to_paths(
    sim_inputs: typing.Sequence[ptypes.SimInput],
    interp_type: pinterp.InterpType = pinterp.InterpType.LINEAR,
) -> typing.Sequence[ptypes.SimInput]:
    """Given a sequence of SimInputs, where the params might be user-specified ParamSpecs, resolve the ParamSpecs to AbstractPaths.

    Args:
        sim_inputs (typing.Sequence[ptypes.SimInput]): _description_
        interp_type (pinterp.InterpType, optional): _description_. Defaults to pinterp.InterpType.LINEAR.

    Returns:
        typing.Sequence[ptypes.SimInput]: a sequence of SimInputs where the params have been resolved to AbstractPaths.
    """

    # Check that the time base is the same size for all SimInputs.
    time_base_sizes = [sim_input.time.size for sim_input in sim_inputs]
    if not all(size == time_base_sizes[0] for size in time_base_sizes):
        raise ValueError(f"Expected all time bases to be the same size. Got sizes: {time_base_sizes}.")

    def resolve_param_spec(sim_input):
        resolved_params = build_param_paths(sim_input.params, sim_input.time, interp_type)
        return ptypes.SimInput(time=sim_input.time, initial_state=sim_input.initial_state, params=resolved_params)

    return [resolve_param_spec(sim_input) for sim_input in sim_inputs]
