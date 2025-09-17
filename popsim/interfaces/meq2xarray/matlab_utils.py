from typing import Any

import numpy as np
import scipy.io as sio
from scipy.sparse import csc_matrix


def loadmat(filename: str) -> dict:
    """Load a .mat file into a dictionary. Courtesy of https://stackoverflow.com/a/29126361/119527, with some modifications.

    Args:
        filename (str): The path to the .mat file.

    Returns:
        dict: The contents of the .mat file.
    """

    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return process_dict_from_matfile(data)


def process_dict_from_matfile(d: dict) -> dict:
    for key, val in d.items():
        if isinstance(val, sio.matlab.mat_struct):
            d[key] = matstruct_to_dict(val)
        elif isinstance(val, dict):
            d[key] = process_dict_from_matfile(val)
        elif is_matstruct_array(val):
            d[key] = array_matstruct_to_list(val)
    return d


def is_matstruct_array(arr: Any) -> bool:
    """Check if a numpy array contains mat_struct objects."""
    if not isinstance(arr, np.ndarray):
        return False
    if arr.size == 0:
        return False
    if arr.dtype != object:
        return False
    if arr.ndim == 0:
        return isinstance(arr.item(), sio.matlab.mat_struct)
    return isinstance(arr[0], sio.matlab.mat_struct)


def convert_sparse(flat_dict: dict):
    """Convet sparse matrices to dense."""
    for k, v in flat_dict.items():
        if isinstance(v, csc_matrix):
            flat_dict[k] = v.toarray()
    return flat_dict


def matstruct_to_dict(matobj: sio.matlab.mat_struct) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            d[strg] = matstruct_to_dict(elem)
        elif is_matstruct_array(elem):
            d[strg] = array_matstruct_to_list(elem)
        else:
            d[strg] = elem
    return d


def array_matstruct_to_list(ndarray: np.ndarray) -> list:
    """
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    """
    if np.ndim(ndarray) == 0:
        # Handle the scalar case
        return ndarray.item()

    def _process_elem(sub_elem):
        if isinstance(sub_elem, sio.matlab.mat_struct):
            return matstruct_to_dict(sub_elem)
        elif is_matstruct_array(sub_elem):
            return array_matstruct_to_list(sub_elem)
        else:
            return sub_elem

    elem_list = [_process_elem(sub_elem) for sub_elem in ndarray]

    return elem_list
