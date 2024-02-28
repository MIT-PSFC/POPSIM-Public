import jax
import xarray as xr
from jaxtyping import Array


def keypath_to_string(keypath) -> str:
    """Convert a keypath to a string.

    Args:
        keypath (_type_): _description_

    Returns:
        str: _description_
    """
    # For some reason, there is often a leading dot in the keypath.
    # Get rid of it.
    return ".".join(str(x).lstrip(".") for x in keypath)


def solution_to_xarray(sol, multi_episode: bool) -> xr.Dataset:
    return time_and_pytree_to_xarray(sol.ts, sol.ys, multi_episode)


def time_and_pytree_to_xarray(time, tree, multi_episode: bool) -> xr.Dataset:
    # Convert a solution to an xarray, supporting 2D arrays for "time" and "episode".
    leaves_with_path = jax.tree_util.tree_leaves_with_path(tree)

    def convert_array(arr: Array) -> Array:
        if multi_episode:
            # TODO(allenw): handle case where there are additional dimensions/coords.
            return (["episode", "time"], arr)
        else:
            return (["time"], arr)

    variables = {keypath_to_string(keypath): convert_array(leaf) for keypath, leaf in leaves_with_path}

    # Expect time to be the same for all episodes.
    assert (time == time[0]).all()

    # Expect all leaves to have the same leading dimension.
    first_leaf_data = leaves_with_path[0][1]
    assert all(leaf.shape[0] == first_leaf_data.shape[0] for _, leaf in leaves_with_path)
    coords = {"time": time[0]}

    if multi_episode:
        coords["episode"] = list(range(first_leaf_data.shape[0]))

    dataset = xr.Dataset(data_vars=variables, coords=coords)
    return dataset
