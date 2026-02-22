import typing
import warnings

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, ArrayLike, PyTree
from loguru import logger
from xarray_jax import var_change_on_unflatten

import popsim.tree_util as ptu
from popsim.array_utils import jax_to_numpy_array

# A tuple of the name of the extra dimension and the coordinates corresponding to that dimension.
# Currently,
ExtraDimAndCoord = tuple[str, Array | np.ndarray]
ExtraDimAndCoordSpec = ExtraDimAndCoord | typing.Sequence[ExtraDimAndCoord] | None

DEFAULT_SIM_DIM_NAME = "simulation"
DEFAULT_TIME_DIM_NAME = "time"


def make_data_array(
    name: str,
    array: Array,
    dims: list[str],
    coords: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Construct an xarray DataArray given the name, array, dimensions, and coordinates.
    If the number of dimensions in the array exceeds the number of specified dims, extra dimensions will be generated with default names.

    Args:
        name (str): name of the resulting DataArray.
        array (Array): the data to wrap in a DataArray.
        dims (list[str]): list of dimension names of the array.
        coords (dict[str, xr.DataArray]): dictionary of coordinates for the DataArray.

    Returns:
        xr.DataArray: the constructed DataArray.
    """
    array = jax_to_numpy_array(array)

    if array.ndim == len(dims):
        # If the number of dimensions in the array matches the number of specified dims, we can just create the DataArray.
        return xr.DataArray(array, dims=dims, coords=coords, name=name)
    elif array.ndim > len(dims):
        # If the array has more dimensions than the number of specified dims, and there was no attempt to specify additional dimensions, we can add extra dimensions with default names.
        n_missing_dims = array.ndim - len(dims)

        logger.debug(
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


def _handle_xr_types(data: xr.DataArray | xr.Variable, base_coords, name) -> xr.DataArray:
    if isinstance(data, xr.Variable):
        da = xr.DataArray(data, coords=base_coords, name=name)
        return da
    elif isinstance(data, xr.DataArray):
        data.name = name
        return data
    elif isinstance(data, xr.Dataset):
        # Add the name as a prefix to the variables in the dataset.
        if name:
            data = data.rename({var: f"{name}_{var}" for var in data.data_vars})
        return data
    else:
        raise ValueError("Only xr.Variable and xr.DataArray are supported.")


def pytree_to_xarray(
    tree: PyTree[Array | xr.Variable | xr.DataArray | xr.Dataset],
    base_dims: list[str] | None = None,
    base_coords: dict[str, xr.DataArray] | None = None,
) -> xr.Dataset:
    """Convert a PyTree of array-likes, xr.Variables, and xr.DataArrays to an xarray Dataset.

    Args:
        tree (PyTree[ArrayLike  |  xr.Variable  |  xr.DataArray | xr.Dataset]): tree containing array-like quantities, xr.Variables, xr.DataArrays, or xr.Datasets.
        base_dims (list[str]): list of dimension names for leaf arrays.
        base_coords (dict[str, xr.DataArray]): coordinates to add to all leaves.

    Returns:
        xr.Dataset: dataset containing the converted PyTree.
    """

    if base_coords is None:
        base_coords = []
    if base_dims is None:
        base_dims = []

    def process_tree_leaf(path, data: np.ndarray | ArrayLike | xr.DataArray | xr.Variable) -> xr.DataArray:
        name = ptu.keypath_to_string(path)
        if isinstance(data, xr.Variable | xr.DataArray | xr.Dataset):
            return _handle_xr_types(data, base_coords, name)
        elif eqx.is_array_like(data):
            data = np.asarray(data) if not eqx.is_array(data) else data
            return make_data_array(name=name, array=jnp.asarray(data), dims=base_dims, coords=base_coords)
        else:
            raise ValueError("Expected ArrayLike, xr.Variable, or xr.DataArray.")

    # Construct a tree of DataArrays.
    paths_and_leaves = jax.tree_util.tree_leaves_with_path(
        tree,
        is_leaf=lambda x: isinstance(x, xr.Dataset | xr.DataArray | xr.Variable | ArrayLike),
    )

    das_and_ds = [process_tree_leaf(path, data) for path, data in paths_and_leaves]

    dataarrays = [da for da in das_and_ds if isinstance(da, xr.DataArray)]
    datasets = [da for da in das_and_ds if isinstance(da, xr.Dataset)]

    ds = xr.merge(datasets)

    ds_from_das = xr.merge(dataarrays)
    ds = xr.merge([ds, ds_from_das])
    return ds


def time_and_pytree_to_xarray(
    time: Array,
    tree: PyTree[Array | xr.Variable | xr.DataArray],
    multi_simulation: bool = False,
) -> xr.Dataset:
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
            dims=(DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME) if time.ndim == 2 else (DEFAULT_TIME_DIM_NAME),
        )
        sim_coord = xr.DataArray(
            np.arange(nsims),
            dims=(DEFAULT_SIM_DIM_NAME),
        )
        base_dims = [DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME]
    else:
        # In the single-simulation case, expect a 1D time array.
        assert time.ndim == 1

        # Build the time and simulation coordinates and the list of dimensions.
        time_coord = xr.DataArray(time, dims=(DEFAULT_TIME_DIM_NAME))
        sim_coord = xr.DataArray(0)
        base_dims = [
            DEFAULT_TIME_DIM_NAME,
        ]

    base_coords = {
        DEFAULT_TIME_DIM_NAME: time_coord,
        DEFAULT_SIM_DIM_NAME: sim_coord,
    }
    return pytree_to_xarray(tree, base_dims, base_coords)


def add_dim_to_vars(tree: PyTree, dim_name: str) -> PyTree:
    """Given a PyTree that may contain xr.Variable instances, add a dimension to the xr.Variables in the leading dimension. An example use case involves running a simulation forward in time, which requires adding a time dimension to the xr.Variable instances.

    Args:
        tree (PyTree): PyTree that may contain xr.Variable instances.
        dim_name (str): Name of the dimension to add.

    Returns:
        PyTree: PyTree with the dimension added to the xr.Variables.
    """

    def var_change_fn(var: xr.Variable):
        var._dims = (dim_name, *var._dims)
        return var

    return jax.tree.map(
        lambda x: var_change_fn(x) if isinstance(x, xr.Variable) else x,
        tree,
        is_leaf=lambda x: isinstance(x, xr.Variable),
    )


def remove_dim_from_vars(tree: PyTree, dim_name: str) -> PyTree:
    """Given a PyTree that may contain xr.Variable instances, remove a dimension from the xr.Variables. An example use case involves applying a vmap across simulation cases, which requires removing the simulation dimension from the xr.Variable instances.

    Args:
        tree (PyTree): PyTree that may contain xr.Variable instances.
        dim_name (str): Name of the dimension to remove.

    Returns:
        PyTree: PyTree with the dimension removed from the xr.Variables.
    """

    def var_change_fn(var: xr.Variable):
        var._dims = tuple(d for d in var._dims if d != dim_name)
        return var

    return jax.tree.map(
        lambda x: var_change_fn(x) if isinstance(x, xr.Variable) else x,
        tree,
        is_leaf=lambda x: isinstance(x, xr.Variable),
    )


def run_function_with_dim_removed(fun: typing.Callable[..., typing.Any], fun_inputs: tuple, dim_remove: int) -> typing.Any:
    """
    Runs a function with a specified dimension removed from its variables, then adds the dimension back
    to the output variables after execution as the leading dimension.

    The primary use case in mind is when we want to, for example, vmap a function across the simulation dimension. In this case, we would like to remove the simulation dimension from the variables before executing the function, then add it back to the output variables after execution.

    Args:
        fun (Callable[..., Any]): The function to execute.
        fun_inputs (tuple): The inputs to pass to `fun`.
        dim_remove (int): The index of the dimension to remove from the variables before executing `fun`.

    Returns:
        Any: The output of `fun` with the removed dimension re-added to the variables.
    """
    with var_change_on_unflatten(lambda var: remove_dim_from_vars(var, dim_remove)):
        out = fun(*fun_inputs)

    out = add_dim_to_vars(out, dim_remove)
    return out


def scramble_xr(obj: xr.DataArray | xr.Dataset | xr.DataTree, zero: bool = False) -> xr.DataArray | xr.Dataset | xr.DataTree:
    """Scramble the data in an xarray object by replacing it with random values or zeros.

    Args:
        obj (typing.Union[xr.DataArray, xr.Dataset, xr.DataTree]): The xarray object to scramble. Can be a DataArray, Dataset, or DataTree.
        zero (bool, optional): If True, replaces all data with zeros. Otherwise, use np.random.random. Defaults to False.

    Returns:
        typing.Union[xr.DataArray, xr.Dataset, xr.DataTree]: The scrambled xarray object.
    """

    def _scrambler(arr):
        if zero:
            return np.zeros(arr.shape, dtype=arr.dtype)
        else:
            return np.random.random(arr.shape)

    # DataArray case: build a new DataArray that preserves everything except the data.
    if isinstance(obj, xr.DataArray):
        return xr.DataArray(
            _scrambler(obj),
            dims=obj.dims,
            coords=obj.coords,
            name=obj.name,
            attrs=obj.attrs,
        )

    # Dataset case: map over the DataArrays in the Dataset and apply the same randomization.
    elif isinstance(obj, xr.Dataset):

        def _rand_da(da: xr.DataArray) -> xr.DataArray:
            return xr.DataArray(
                _scrambler(da),
                dims=da.dims,
                coords=da.coords,
                name=da.name,
                attrs=da.attrs,
            )

        return obj.map(_rand_da)

    # DataTree case: map over the datasets in the DataTree and apply the same randomization.
    elif xr.DataTree is not None and isinstance(obj, xr.DataTree):
        # map_over_datasets expects a function that takes a Dataset and returns a Dataset
        return obj.map_over_datasets(lambda x: scramble_xr(x, zero=zero))

    else:
        raise TypeError(f"Unsupported type {type(obj)}; expected xr.DataArray, xr.Dataset, or xarray.DataTree")
