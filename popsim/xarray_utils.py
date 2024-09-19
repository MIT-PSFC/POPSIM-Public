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
    base_dims: list[str],
    base_coords: dict[str, xr.DataArray],
    extra_dim_and_coord_spec: ExtraDimAndCoordSpec,
) -> xr.DataArray:
    extra_dims, extra_coords = make_extra_dims_and_coords(extra_dim_and_coord_spec)
    specified_dims = base_dims + extra_dims
    coords = base_coords | extra_coords

    array = jax_to_numpy_array(array)

    if array.ndim == len(specified_dims):
        # If the number of dimensions in the array matches the number of specified dims, we can just create the DataArray.
        return xr.DataArray(array, dims=specified_dims, coords=coords, name=name)
    elif array.ndim > len(specified_dims) and len(extra_dims) == 0:
        # If the array has more dimensions than the number of specified dims, and there was no attempt to specify additional dimensions, we can add extra dimensions with default names.
        n_missing_dims = array.ndim - len(specified_dims)

        logger.info(
            f"Variable {name} has more dimensions, {array.ndim}, than the number of specified dimensions, {len(specified_dims)}. {n_missing_dims} extra dimensions will be generated with names {name}_extra_dim_i."
        )

        # Auto-generate names for the extra dimensions.
        extra_generated_dims = [f"{name}_extra_dim_{i}" for i in range(n_missing_dims)]

        all_dims = specified_dims + extra_generated_dims
        return xr.DataArray(array, dims=all_dims, coords=coords, name=name)
    else:
        warnings.warn(
            f"Skipping variable {name} due to mismatch between the number of dimensions of the array ({array.ndim}) and the number of specified dimensions ({len(specified_dims)}).",
            stacklevel=2,
        )
        return None


def tree_dim_and_coords_to_xarray(
    tree: PyTree[Array],
    base_dims: list[str],
    base_coords: dict[str, xr.DataArray],
    extra_dim_and_coord_tree: PyTree[typing.Optional[ExtraDimAndCoordSpec]],
) -> xr.Dataset:
    def _make_data_array(keypath, array, extra_dim_and_coord_spec):
        name = ptu.keypath_to_string(keypath)
        return make_data_array(
            name=name, array=array, base_dims=base_dims, base_coords=base_coords, extra_dim_and_coord_spec=extra_dim_and_coord_spec
        )

    # Construct a tree of DataArrays.
    da_tree = jax.tree_util.tree_map_with_path(_make_data_array, tree, extra_dim_and_coord_tree)

    # Get a list of the resulting DataArrays.
    dataarrays = jax.tree_util.tree_leaves(da_tree, is_leaf=lambda x: isinstance(x, xr.DataArray))

    # Convert to a Dataset.
    ds = xr.merge(dataarrays)
    return ds


def make_extra_dims_and_coords(coord_spec: ExtraDimAndCoordSpec) -> tuple[list[str], dict[str, xr.DataArray]]:
    """Given a specification of extra dimensions and coordinates, return the list of extra dimensions and a dictionary of extra coordinates in the form of xarray DataArrays.

    Args:
        coord_spec (ExtraDimAndCoordSpec): A specification of extra dimensions and coordinates.

    Raises:
        ValueError: If coord_spec is not a tuple, list or None.

    Returns:
        tuple[list[str], dict[str, xr.DataArray]]: first element is a list of extra dimensions, second element is a dictionary of extra coordinates.
    """
    if coord_spec is None:
        return [], {}
    elif isinstance(coord_spec, tuple):
        name, coord = coord_spec
        return [name], {
            name: xr.DataArray(coord, dims=[name], name=name),
        }
    elif isinstance(coord_spec, list):
        extra_dims = [name for name, _ in coord_spec]
        extra_coords = {name: xr.DataArray(coord, dims=[name], name=name) for name, coord in coord_spec}
        return extra_dims, extra_coords
    else:
        raise ValueError(f"coord_spec must be a tuple, list or None, got {type(coord_spec)}.")


def solution_to_xarray(sol: diffrax.Solution, coord_tree: PyTree = None, multi_simulation: bool = False) -> xr.Dataset:
    """Convert a diffrax.Solution to an xarray. Thin wrapper around time_and_pytree_to_xarray.

    Args:
        sol (diffrax.Solution): diffrax.Solution object.
        coord_tree (PyTree[typing.Optional[list[Coords]]]): shadowing the tree, a PyTree of lists of tuples containing coordinate specifications.
                                                            Note the ith element of the list corresponds to the ith dimension of the array.

        multi_simulation (bool): Whether the solution contains multiple simulations.

    Returns:
        xr.Dataset: An xarray dataset.
    """
    return time_and_pytree_to_xarray(sol.ts, sol.ys, coord_tree, multi_simulation)


def time_and_pytree_to_xarray(
    time: Array, tree: PyTree[Array], extra_dim_and_coord_tree: PyTree[ExtraDimAndCoordSpec] = None, multi_simulation: bool = False
) -> xr.Dataset:
    """Convert a time array and a PyTree of arrays to an xarray dataset. In the multi-simulation case, assign numbered names to the simulations.

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
        base_dimensions = ["simulation", "time"]
    else:
        # In the single-simulation case, expect a 1D time array.
        assert time.ndim == 1

        # Build the time and simulation coordinates and the list of dimensions.
        time_coord = xr.DataArray(time, dims=("time"))
        sim_coord = xr.DataArray(0)
        base_dimensions = [
            "time",
        ]

    base_coords = {
        "time": time_coord,
        "simulation": sim_coord,
    }

    # Construct the coordnates tree.
    extra_dim_and_coord_tree = jax.tree.map(lambda _: None, tree) if extra_dim_and_coord_tree is None else extra_dim_and_coord_tree

    # Construct the dataset.
    ds = tree_dim_and_coords_to_xarray(
        tree, base_dims=base_dimensions, base_coords=base_coords, extra_dim_and_coord_tree=extra_dim_and_coord_tree
    )
    return ds
