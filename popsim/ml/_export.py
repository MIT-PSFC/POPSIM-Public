"""
Utilities for exporting models.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import jax
import xarray as xr

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import DEFAULT_SAMPLE_DIM, DataLoader
from popsim.ml.eval import EvalData, eval_model_on_data
from popsim.tree_util import to_json_compatible


def write_json(module: TrainableModel, path: str | Path):
    """Write a trainable module to a json file.
    The output json file includes:
        1) A string representation of the tree structure of the module.
        2) The parameters of the module in a json-compatible format.

    Args:
        module (TrainableModel): the module to write to a json file.
        path (str | Path): the path to write the json file to.
    """
    path = Path(path)
    assert path.suffix == ".json"

    tree_dict = to_json_compatible(module)
    treedef = str(jax.tree.structure(module))
    static_fields = get_static_fields(module)

    out = {
        "static_fields": static_fields,
        "model_parameters": tree_dict,
        "treedef": treedef,
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=4)


def get_static_fields(module: dataclass) -> dict[str, Any]:
    """
    Recursively retrieves all static fields from a dataclass.

    Args:
        module (dataclass): The dataclass instance to inspect.

    Returns:
        dict: A dictionary containing field names and their values for static fields.
    """
    if not is_dataclass(module):
        raise TypeError("Expected a dataclass instance")

    static_fields = {}

    for f in fields(module):
        value = getattr(module, f.name)
        if is_dataclass(value):  # If the field is another dataclass, recurse
            nested_static = get_static_fields(value)
            static_fields[f.name] = nested_static
        elif isinstance(value, Sequence) and all(is_dataclass(v) for v in value):
            # JSON only supports lists/arrays, so convert all sequences into a list.
            static_fields[f.name] = [get_static_fields(v) for v in value]
        elif f.metadata.get("static", False):
            static_fields[f.name] = to_json_compatible(value)
        else:
            # Ignore non-static fields
            pass
    return static_fields


def export(module: TrainableModel, path: str | Path, dataloader: DataLoader):
    """Export a trainable module as a JSON. Also generate input and output data as a netCDF file for validation.

    Args:
        module (TrainableModel): the module to export.
        path (str | Path): the path to write the json + netCDF files to.
        dataloader (DataLoader): the dataloader to use to generate validation data.
    """
    # If a dataloader is provided, the path should be a directory
    path = Path(path)

    if not os.path.exists(path):
        os.makedirs(path)

    assert path.is_dir()

    json_file_path = path / "model.json"
    write_json(module, json_file_path)

    eval_data: EvalData = eval_model_on_data(module, dataloader)

    input_ds = eval_data.input_ds
    output_ds = eval_data.output_ds

    # Rename variables in input_ds and output_ds to include "input." and "output."
    input_ds = input_ds.rename_vars({var: f"input.{var}" for var in input_ds.data_vars})
    output_ds = output_ds.rename_vars({var: f"output.{var}" for var in output_ds.data_vars})

    ds = xr.merge([input_ds, output_ds])

    ds = ds.reset_index(DEFAULT_SAMPLE_DIM)

    ds.to_netcdf(path / "eval_data.nc")
