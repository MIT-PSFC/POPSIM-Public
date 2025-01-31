import typing

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

import popsim.interp as pinterp
import popsim.types as ptypes
from popsim.sim_utils import SimInput


def build_param_paths(
    inputs: ptypes.Inputspec, time_base: Array, interp_type: pinterp.InterpType = pinterp.InterpType.LINEAR
) -> PyTree[diffrax.AbstractPath]:
    """Given a user-specified inputs specification that is a PyTree of "ConstantOrPathSpec", generate
    a new PyTree of "AbstractPath" on the given time_base. Leaves are handeled as follows:

        1) If the leaf is a dictionary with float keys, it is assumed to be a PathSpec and is interpolated onto "time_base".
        2) If the leaf is an instance of AbstractPath, we re-map to "time_base" and interpolate again.
        3) Otherwise, the leaf is assumed to be a constant, is repeated for all times in "time_base", and interpolated.

    Why interpolate everything? This is done to ensure the inputs to jax.jitted functions are all the same shape across
    calls to prevent re-compiling (which is the main computational cost right now). This way, if the user switches from
    specifying a inputs as a constant to trajectory, the function will not need to be re-compiled.

    Args:
        inputs (ptypes.Inputspec): A user-specified param specification.
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

    return jax.tree.map(interp_onto_timebase, inputs, is_leaf=lambda x: is_path_spec(x) or isinstance(x, diffrax.AbstractPath))


def param_specs_to_paths(
    sim_inputs: typing.Sequence[SimInput],
    interp_type: pinterp.InterpType = pinterp.InterpType.LINEAR,
) -> typing.Sequence[SimInput]:
    """Given a sequence of SimInputs, where the inputs might be user-specified Inputspecs, resolve the Inputspecs to AbstractPaths.

    Args:
        sim_inputs (typing.Sequence[SimInput]): _description_
        interp_type (pinterp.InterpType, optional): _description_. Defaults to pinterp.InterpType.LINEAR.

    Returns:
        typing.Sequence[SimInput]: a sequence of SimInputs where the inputs have been resolved to AbstractPaths.
    """

    # Check that the time base is the same size for all SimInputs.
    time_base_sizes = [sim_input.time.size for sim_input in sim_inputs]
    if not all(size == time_base_sizes[0] for size in time_base_sizes):
        raise ValueError(f"Expected all time bases to be the same size. Got sizes: {time_base_sizes}.")

    # Check that the time bases are monotonic.
    for sim_input in sim_inputs:
        if not jnp.all(jnp.diff(sim_input.time) >= 0):
            raise ValueError("Time base must be monotonic.")

    def resolve_param_spec(sim_input):
        resolved_inputs = build_param_paths(sim_input.inputs, sim_input.time, interp_type)
        return SimInput(time=sim_input.time, initial_state=sim_input.initial_state, inputs=resolved_inputs)

    return [resolve_param_spec(sim_input) for sim_input in sim_inputs]
