import typing
import warnings

import diffrax
import jax
import numpy as np
import xarray as xr
from jaxtyping import Array, PyTree
from loguru import logger

import popsim.tree_util as ptu
from popsim.array_utils import jax_to_numpy_array

# A tuple of the name of the extra dimension and the coordinates corresponding to that dimension.
# Currently,
ExtraDimAndCoord = tuple[str, typing.Union[Array, np.ndarray]]
ExtraDimAndCoordSpec = typing.Union[ExtraDimAndCoord, typing.Sequence[ExtraDimAndCoord], None]


def make_data_array(
    name: str,
    array: Array,
    dims: list[str],
    coords: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Construct an xarray DataArray given the name, array, dimensions, and coordinates.
    If the number of dimensions in the array exceeds the number of specified dims, extra dimensions will be generated with default names.

    Args:
        name (str): _description_
        array (Array): _description_
        dims (list[str]): _description_
        coords (dict[str, xr.DataArray]): _description_

    Returns:
        xr.DataArray: _description_
    """
    array = jax_to_numpy_array(array)

    if array.ndim == len(dims):
        # If the number of dimensions in the array matches the number of specified dims, we can just create the DataArray.
        return xr.DataArray(array, dims=dims, coords=coords, name=name)
    elif array.ndim > len(dims):
        # If the array has more dimensions than the number of specified dims, and there was no attempt to specify additional dimensions, we can add extra dimensions with default names.
        n_missing_dims = array.ndim - len(dims)

        logger.info(
            f"Variable {name} has more dimensions, {array.ndim}, than the number of specified dimensions, {len(dims)}. {dims} extra dimensions will be generated with names {name}_extra_dim_i."
        )

        # Auto-generate names for the extra dimensions.
        extra_generated_dims = [f"{name}_extra_dim_{i}" for i in range(n_missing_dims)]
        return xr.DataArray(array, dims=dims + extra_generated_dims, coords=coords, name=name)
    else:
        warnings.warn(
            f"Skipping variable {name} due to mismatch between the number of dimensions of the array ({array.ndim}) and the number of specified dimensions ({len(dims)}).",
            stacklevel=2,
        )
        return None


def solution_to_xarray(sol: diffrax.Solution, multi_simulation: bool = False) -> xr.Dataset:
    """Convert a diffrax.Solution to an xarray. Thin wrapper around time_and_pytree_to_xarray.

    Args:
        sol (diffrax.Solution): diffrax.Solution object.
        multi_simulation (bool): Whether the solution contains multiple simulations.

    Returns:
        xr.Dataset: An xarray dataset.
    """
    return time_and_pytree_to_xarray(sol.ts, sol.ys, multi_simulation)


def _handle_xr_types(data: xr.DataArray | xr.Variable, base_dims, base_coords, name) -> xr.DataArray:
    """A bit of a weird hack to account for the fact that running simulations currently does not add the dimensions corresponding to the simulation and time to the xarray objects. This function adds them."""
    if isinstance(data, xr.Variable):
        data._dims = (*base_dims, *data._dims)
        da = xr.DataArray(data, coords=base_coords, name=name)
        return da
    elif isinstance(data, xr.DataArray):
        data.variable._dims = (*base_dims, *data.variable._dims)
        data = data.assign_coords(base_coords)
        data.name = name
        return data
    else:
        raise ValueError("xr.Variable, or xr.DataArray.")


def time_and_pytree_to_xarray(time: Array, tree: PyTree[Array | xr.Variable | xr.DataArray], multi_simulation: bool = False) -> xr.Dataset:
    """Convert a time array and a PyTree of arrays, xr.Variable, and xr.DataArray instances to a xr.Dataset. In the multi-simulation case, assign numbered names to the simulations.

    Args:
        time (Array): time array. In the multi-simulation case, this can be either a 1D or 2D array. In the single-simulation case, this is a 1D array.
        tree (PyTree[Array]): PyTree of arrays. In the multi-simulation case, expect shapes of (simulation, time, ...). In the single-simulation case, expect (time, ...).
        extra_dim_and_coord_tree (PyTree[ExtraCoordSpec]): PyTree of extra dimensions and coordinates for each array in the tree.
        multi_simulation (bool): whether the time and tree are multi-simulation.

    Returns:
        xr.Dataset: An xarray dataset.
    """

    if multi_simulation:
        assert time.ndim in (2, 1)

        # If the time arrays are the same for all simulations, we can just use the first one.
        if time.ndim == 2 and np.all(time == time[0]):
            time = time[0]

        # Check that every array has the same number of simulations.
        nsims_tree = jax.tree.map(lambda x: x.shape[0], tree)
        tree_leaves = jax.tree.leaves(nsims_tree)
        if len(set(tree_leaves)) > 1:
            raise ValueError("In multi-simulation mode, all arrays must have the same number of simulations.")
        nsims = tree_leaves[0]

        # Build the time and simulation coordinates and the list of dimensions.
        time_coord = xr.DataArray(
            time,
            dims=("simulation", "time") if time.ndim == 2 else ("time"),
        )
        sim_coord = xr.DataArray(
            np.arange(nsims),
            dims=("simulation"),
        )
        base_dims = ["simulation", "time"]
    else:
        # In the single-simulation case, expect a 1D time array.
        assert time.ndim == 1

        # Build the time and simulation coordinates and the list of dimensions.
        time_coord = xr.DataArray(time, dims=("time"))
        sim_coord = xr.DataArray(0)
        base_dims = [
            "time",
        ]

    base_coords = {
        "time": time_coord,
        "simulation": sim_coord,
    }

    def process_tree_leaf(path, data: np.ndarray | Array | xr.DataArray | xr.Variable) -> xr.DataArray:
        name = ptu.keypath_to_string(path)
        if isinstance(data, (np.ndarray, Array)):
            return make_data_array(name=name, array=data, dims=base_dims, coords=base_coords)
        elif isinstance(data, (xr.Variable, xr.DataArray)):
            return _handle_xr_types(data, base_dims, base_coords, name)
        else:
            raise ValueError("Expected Array, xr.Variable, or xr.DataArray.")

    # Construct a tree of DataArrays.
    paths_and_leaves = jax.tree_util.tree_leaves_with_path(tree, is_leaf=lambda x: isinstance(x, (xr.DataArray, xr.Variable, Array)))

    dataarrays = [process_tree_leaf(path, data) for path, data in paths_and_leaves]

    ds = xr.merge(dataarrays)
    return ds
