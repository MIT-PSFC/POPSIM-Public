from enum import Enum, IntEnum

import jax
import xarray as xr
from jaxtyping import Array

import popsim.tree_util as ptu


def keypath_to_string(keypath) -> str:
    """Convert a keypath to a string.

    Args:
        keypath (_type_): _description_

    Returns:
        str: _description_
    """

    def string_func(x):
        key = ptu.get_key(x)
        if isinstance(x, (Enum, IntEnum)):
            return key.name
        return str(key)

    strings = [string_func(x) for x in keypath]

    # For some reason, there is often a leading dot in the keypath.
    strings = [x.lstrip(".") for x in strings]

    # The gui also does show quantities between <>.
    strings = [x.replace("<", "").replace(">", "") for x in strings]

    # Strip away brackets.
    strings = [x.replace("[", "").replace("]", "") for x in strings]

    # Finally strip away unnecessary quotes.
    strings = [x.replace("'", "") for x in strings]

    return ".".join(strings)


def solution_to_xarray(sol, multi_simulation: bool) -> xr.Dataset:
    return time_and_pytree_to_xarray(sol.ts, sol.ys, multi_simulation)


def time_and_pytree_to_xarray(time, tree, multi_simulation: bool, rhogrid: Array = None) -> xr.Dataset:
    # Convert a solution to an xarray, supporting 2D arrays for "time" and "simulation".
    leaves_with_path = jax.tree_util.tree_leaves_with_path(tree)

    def convert_array(arr: Array) -> Array:
        if multi_simulation:
            if arr.ndim == 2:
                return (["simulation", "time"], arr)
            if arr.ndim == 3:
                return (["simulation", "time", "rho"], arr)
            else:
                raise ValueError(f"Array has unexpected shape {arr.shape}.")
        else:
            if arr.ndim == 1:
                return (["time"], arr)
            if arr.ndim == 2:
                return (["time", "rho"], arr)
            else:
                raise ValueError(f"Array has unexpected shape {arr.shape}.")

    variables = {keypath_to_string(keypath): convert_array(leaf) for keypath, leaf in leaves_with_path}

    # Expect time to be the same for all simulations.
    if multi_simulation:
        assert (time == time[0]).all()

    # Expect all leaves to have the same leading dimension.
    first_leaf_data = leaves_with_path[0][1]
    assert all(leaf.shape[0] == first_leaf_data.shape[0] for _, leaf in leaves_with_path)
    coords = {"time": time[0]} if multi_simulation else {"time": time}

    if multi_simulation:
        coords["simulation"] = list(range(first_leaf_data.shape[0]))
    if rhogrid is not None:
        coords["rho"] = rhogrid

    dataset = xr.Dataset(data_vars=variables, coords=coords)
    return dataset
