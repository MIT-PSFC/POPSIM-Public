from collections import defaultdict

import numpy as np
import xarray as xr
from loguru import logger

from popsim.interfaces.meq2xarray.dim_map import VARS_TO_DIM_MAP
from popsim.interfaces.meq2xarray.matlab_utils import convert_sparse, loadmat
from popsim.utils import flatten_dict

"""
Collection of utilities to convert meq .mat files containing L, LY, and LX structs (and other structures) into xarray Datasets.

Some of the utilities are TCV specific, ask @allenw.
"""


def tcv_db_to_xr(path: str) -> xr.DataTree:
    """Give a path to a mat file containing TCV data (ask allenw), convert it to an xarray DataTree.

    Args:
        path (str): The path to the .mat file.

    Returns:
        xr.DataTree: The xarray DataTree containing the data from the .mat file.
    """
    data = loadmat(path)

    # We want time to be the coordinate, but time_idx to be the dimension to allow for
    # datasets that have shots with different time bases.
    fbt = meqstructs2xarray(data["fbt_data"]).swap_dims({"time": "time_idx"}).expand_dims("shot")
    liuqe = meqstructs2xarray(data["liuqe_data"]).swap_dims({"time": "time_idx"}).expand_dims("shot")
    ss = process_ss(data["state_space_systems"]).swap_dims({"time": "time_idx"}).expand_dims("shot")
    acts = process_acts(data["acts"]).swap_dims({"time": "time_idx"}).expand_dims("shot")

    return xr.DataTree.from_dict({"fbt": fbt, "liuqe": liuqe, "ss": ss, "acts": acts})


def tcv_fbt_to_xr(path: str) -> xr.Dataset:
    """Given a path to a TCV FBT .mat file, containing L, LY, and LX structs, convert it to an xarray Dataset.

    Args:
        path (str): The path to the .mat file.

    Returns:
        xr.Dataset: The xarray Dataset containing the data from the .mat file.
    """
    return meqstructs2xarray(loadmat(path)).swap_dims({"time": "time_idx"}).expand_dims("shot")


def meqstructs2xarray(mat_contents: dict, debug: bool = False, non_arrays_as_attrs: bool = False) -> xr.Dataset:
    """Given a dictionary containing L, LY, and possibly LX structs load it into an xarray Dataset.
    Variables with known dimension structure as defined in DIM_TO_VARS_MAP are loaded into the xr.Dataset.
    If there are additional variables you want to load, you can use the debug flag to get the variables with unknown
    dimensions.

    Args:
        meqmat_file (typing.Union[str, Path]): The path to the meq .mat file.
        debug (bool, optional): If True, return the dataset, unknown dataarrays, and the original data dictionary. Defaults to False.
        non_arrays_as_attrs (bool, optional): If True, non-array variables are added as attributes to the dataset. Defaults to False.

    Returns:
        xr.Dataset: The xarray Dataset containing the data from the meq file.
    """

    data_dict = {}
    # Check that both L and LY are present
    if "L" not in mat_contents:
        raise ValueError("L not found in .mat file")
    if "LY" not in mat_contents:
        raise ValueError("LY not found in .mat file")

    L_flat = flatten_dict(mat_contents["L"], parent_key="L", sep=".")
    data_dict = data_dict | L_flat

    LY_flat = flatten_dict(mat_contents["LY"], parent_key="LY", sep=".")
    data_dict = data_dict | LY_flat

    if "LX" in mat_contents:
        LX_flat = flatten_dict(mat_contents["LX"], parent_key="LX", sep=".")
        data_dict = data_dict | LX_flat

    # Convert sparse matrices to dense
    data_dict = convert_sparse(data_dict)

    coords = build_coordinates(data_dict)

    ds = xr.Dataset(coords=coords)

    non_arrays = {k: v for k, v in data_dict.items() if not isinstance(v, np.ndarray)}

    if non_arrays_as_attrs:
        ds.attrs = non_arrays

    def _handle(item):
        if isinstance(item, list):
            return np.array(item)
        elif isinstance(item, np.ndarray):
            return item
        else:
            return None

    # Convert lists to numpy arrays
    arrays = {k: _handle(v) for k, v in data_dict.items() if isinstance(v, list | np.ndarray)}

    # Remove items that have been added to the dataset as coordinates
    arrays = {k: v for k, v in arrays.items() if k not in coords.keys()}

    known_das = make_known_datarrays(arrays)

    # Dataset.update mutates in place and returns None (xarray >= 2026).
    ds.update(known_das)

    unknown_arrays = {k: v for k, v in arrays.items() if k not in known_das.keys()}

    inferred_das = {k: _auto_array_to_dims(v, ds) for k, v in unknown_arrays.items()}

    inferred_das = {k: v for k, v in inferred_das.items() if v is not None}

    ds.update(inferred_das)

    unknown_das = {k: v for k, v in unknown_arrays.items() if k not in inferred_das.keys()}

    # Assign a time_idx dimension to the dataset
    # This will allow us to concatenate shots with different time lengths
    ds = ds.assign_coords(time_idx=("time", range(len(ds["time"]))))

    shot = non_arrays.get("L.P.shot", None)
    ds = ds.assign_coords(shot=shot)

    if debug:
        return ds, unknown_das, data_dict
    else:
        return ds


def make_known_datarrays(data_dict: dict) -> dict[str, xr.DataArray]:
    """Given a dictionary of arrays, for all arrays that have dimensions specified in VARS_TO_DIM_MAP, create a DataArray.

    Args:
        data_dict (dict): A dictionary of arrays.

    Returns:
        dict[str, xr.DataArray]: A dictionary of DataArrays.
    """
    das = {}
    for k, v in data_dict.items():
        if k in VARS_TO_DIM_MAP and v.size > 0:
            try:
                dims = VARS_TO_DIM_MAP[k]
                das[k] = xr.DataArray(v, dims=dims, name=k)
            except ValueError as e:
                logger.error(f"Error creating DataArray for {k}: {e}")
    return das


def build_coordinates(data_dict: dict) -> dict[str, xr.DataArray]:
    """Given a dictionary of L, LY, and LX structs, build the coordinates for the xarray Dataset.

    Args:
        data_dict (dict): The dictionary containing the L, LY, and LX structs.

    Returns:
        dict[str, xr.DataArray]: A dictionary of DataArrays containing the coordinates.
    """
    time_coord = xr.DataArray(data_dict["LY.t"], dims=["time"], name="time")
    act_labels = xr.DataArray(list(data_dict["L.G.dima"]), dims=["active_coils"], name="active_coils")
    dimv = xr.DataArray(list(data_dict["L.G.dimv"]), dims=["vessel"], name="vessel")
    dimu = xr.DataArray(list(data_dict["L.G.dimu"]), dims=["passive_currents"], name="passive_currents")
    flux_loop = xr.DataArray(list(data_dict["L.G.dimf"]), dims=["flux_loop"], name="flux_loop")
    mag_probe = xr.DataArray(list(data_dict["L.G.dimm"]), dims=["mag_probe"], name="mag_probe")
    pq = xr.DataArray(
        data_dict["L.pq"],
        dims=["pq"],
        name="pq",
        attrs={"description": "user defined radial grid."},
    )
    pQ = xr.DataArray(
        data_dict["L.pQ"],
        dims=["pQ"],
        name="pQ",
        attrs={"description": "user defined radial grid plus 1."},
    )
    oq = xr.DataArray(
        data_dict["L.oq"],
        dims=["oq"],
        name="oq",
        attrs={"description": "poloidal angle theta grid"},
    )
    rx = xr.DataArray(
        data_dict["L.rx"].squeeze(),
        dims=["rx"],
        name="rx",
        attrs={"description": "r coordinates of the 'x' grid"},
    )
    zx = xr.DataArray(
        data_dict["L.zx"].squeeze(),
        dims=["zx"],
        name="zx",
        attrs={"description": "z coordinates of the 'x' grid"},
    )
    ry = xr.DataArray(
        data_dict["L.ry"],
        dims=["ry"],
        name="ry",
        attrs={"description": "r coordinates of the 'y' grid"},
    )
    zy = xr.DataArray(
        data_dict["L.zy"],
        dims=["zy"],
        name="zy",
        attrs={"description": "z coordinates of the 'y' grid"},
    )
    xpoint = xr.DataArray(
        np.arange(data_dict["LY.rX"].shape[0]),
        dims=["xpoint"],
        name="xpoint",
        attrs={"description": "x-point number"},
    )

    shot = np.unique(data_dict["L.P.shot"]).item() if isinstance(data_dict["L.P.shot"], np.ndarray) else data_dict["L.P.shot"]

    shot = xr.DataArray(
        [shot],
        dims=["shot"],
        name="shot",
        attrs={"description": "shot number"},
    )

    coords = [
        time_coord,
        act_labels,
        dimv,
        dimu,
        flux_loop,
        mag_probe,
        pq,
        pQ,
        oq,
        rx,
        zx,
        ry,
        zy,
        shot,
        xpoint,
    ]
    coords = {coord.name: coord for coord in coords}
    return coords


def _auto_array_to_dims(arr: np.ndarray, ds: xr.Dataset) -> xr.Variable | None:
    """Given an array and a dataset, try to identify what dimensions map to the array.

    Return None if the dimensions are ambiguous or don't exist in the dataset.

    Args:
        arr (np.ndarray): The array to auto-identify dimensions for.
        ds (xr.Dataset): The dataset to use for dimension identification.

    Returns:
        Optional[xr.Variable]: The variable with the identified dimensions, or None if the dimensions are ambiguous or don't exist in the dataset.
    """

    # Get the dimension sizes from ds
    dim_sizes = ds.sizes  # dict-like, mapping from dim names to sizes

    # Build a mapping from size to dimension names
    size_to_dims = defaultdict(list)
    for dim_name, size in dim_sizes.items():
        size_to_dims[size].append(dim_name)

    # Now for each size, check if it's unique
    unique_size_to_dim = {size: dims[0] for size, dims in size_to_dims.items() if len(dims) == 1}

    # Now for each axis of arr, get the size
    arr_shape = arr.shape

    # For each axis in arr, we need to find the dimension name in ds that matches the size, if unique
    dim_names = []
    for axis_size in arr_shape:
        if axis_size in unique_size_to_dim:
            dim_name = unique_size_to_dim[axis_size]
            dim_names.append(dim_name)
        else:
            # Size is ambiguous or doesn't exist in ds dimensions
            dim_names.append(None)
    if None in dim_names or len(dim_names) != len(set(dim_names)):
        return None
    else:
        return xr.Variable(dims=dim_names, data=arr)


def process_ss(ss: dict | list[dict]) -> xr.Dataset:
    """Convert state space systems generated by meq to xarray Dataset.

    Args:
        ss (dict | list[dict]): A dictionary or list of dictionaries containing state space systems.

    Returns:
        xr.Dataset: The xarray Dataset containing the state space systems.
    """

    def _process_single(ss):
        A = xr.DataArray(ss["A"], dims=["state_out", "state"])
        B = xr.DataArray(ss["B"], dims=["state_out", "input"])
        C = xr.DataArray(ss["C"], dims=["output", "state"])
        D = xr.DataArray(ss["D"], dims=["output", "input"])
        x0L = xr.DataArray(ss["x0L"], dims=["state"], attrs={"description": "Linearization point in state space"})
        u0L = xr.DataArray(ss["u0L"], dims=["input"], attrs={"description": "Linearization point in input space"})
        yo = xr.DataArray(ss["yo"], dims=["output"], attrs={"description": "Output feed-forwards when using absolute coordinates"})
        y0 = xr.DataArray(
            ss["y0"],
            dims=["output"],
            attrs={"description": "Output feed-forwards when using coordinates relative to linearization point (x0L, u0L)"},
        )
        x0dot = xr.DataArray(ss["x0dot"], dims=["state"], attrs={"description": "Derivative of state at linearization point"})

        state_labels = xr.Variable(dims="state", data=ss["StateName"])
        input_labels = xr.Variable(dims="input", data=ss["InputName"])
        output_labels = xr.Variable(dims="output", data=ss["OutputName"])
        time = xr.Variable(dims="time", data=np.atleast_1d(ss["t"]))
        ds = xr.Dataset(
            {"A": A, "B": B, "C": C, "D": D, "x0L": x0L, "u0L": u0L, "y0": y0, "x0dot": x0dot, "yo": yo},
            coords={"state": state_labels, "state_out": state_labels, "input": input_labels, "output": output_labels, "time": time},
        )
        return ds

    if isinstance(ss, dict):
        return _process_single(ss)
    elif isinstance(ss, list):
        ss_valid = [s for s in ss if isinstance(s, dict)]
        return xr.concat([_process_single(s) for s in ss_valid], dim="time")
    else:
        raise ValueError(f"Expected dict or list of dicts, got {type(ss)}")


def process_acts(acts: dict) -> xr.Dataset:
    """Given the actuators struct from TCV hybrid, convert it to an xarray Dataset.

    Args:
        acts (dict): The actuators struct from TCV hybrid.

    Returns:
        xr.Dataset: The xarray Dataset containing the actuators data.
    """
    time = xr.DataArray(acts["time"], dims=["time"])

    # Get index of the GAS signal.
    dims = acts["dim"].tolist()

    gas_idx = dims.index("GAS")

    active_coils = np.delete(dims, gas_idx)

    d = acts["data"]

    gas_data = d[gas_idx, :]
    non_gas_data = np.delete(d, gas_idx, axis=0)

    gas = xr.DataArray(gas_data, dims=["time"])
    Va = xr.DataArray(non_gas_data, dims=["active_coils", "time"], coords={"active_coils": active_coils})

    ds = xr.Dataset({"Va": Va, "gas": gas, "time": time})
    return ds


def process_sc(sc_dict: dict) -> xr.Dataset:
    """Given the shape control struct generated by LY2SC, convert it to an xarray Dataset.

    Args:
        scs (dict): The shape control struct.

    Returns:
        xr.Dataset: The xarray Dataset containing the shape control data.
    """

    def process_item(k, v):
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return xr.DataArray(data=v, dims=("nc", "time"), name="SC." + k)
        else:
            return None

    das = [process_item(k, v) for k, v in sc_dict.items()]

    das = [x for x in das if x is not None]

    ds = xr.merge(das)
    return ds
