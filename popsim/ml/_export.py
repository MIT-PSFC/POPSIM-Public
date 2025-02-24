"""
Utilities for exporting models.
"""
import json
import os
from pathlib import Path

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

    out = {"treedef": treedef, "model_parameters": tree_dict}

    with open(path, "w") as f:
        json.dump(out, f, indent=4)


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

    ds.to_netcdf(path / "validation_data.nc")
