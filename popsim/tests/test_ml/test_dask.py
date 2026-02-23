import os
import tempfile

import numpy as np
import jax.numpy as jnp
import xarray as xr

from popsim import DATA_DIR
from popsim.ml.dataloading import make_time_dep_dataloader, make_time_indep_dataloader
from popsim.ml.launch import launch_train
from popsim.ml.train_config import TrainConfig
from popsim.modules.transport_predictor.train_configs import BASE_CONFIG as TRANSPORT_PREDICTOR_BASE_CONFIG, update_submodule_configs


def test_dask_time_dep_dataloader():
    # Test that the time-dependent dataloader properly handles dask arrays and only brings them into memory as needed for JAX
    # JAX only works with in-memory arrays, see https://github.com/google-deepmind/graphcast/issues/89
    ds_path_nc =  os.path.join(DATA_DIR, "tcv", "scrambled_transport_sample.nc")
    ds = xr.open_dataset(ds_path_nc, chunks="auto")
    ds = ds.set_coords("time")

    # Ensure we start with a dask array
    assert ds["B0"].chunks is not None, "Dataset should be a dask array with chunks"

    dl = make_time_dep_dataloader(
        ds,
        time_coord="time",
        episode_coord="shot",
        state_init_vars=["Wtot_MJ"],
        input_vars=["B0"],
        target_vars=["Wtot_MJ"],
        nan_handling="drop_slice_all",
        convert_xr_to_jnp=False,  # Don't convert to jnp arrays, since we're testing the dataloader's handling of dask arrays
        batch_size=2, # Small batch size to ensure we're actually only loading one batch at a time
    )

    # Assert that the dataloader's dataset is still a dask array
    assert dl.ds["B0"].chunks is not None, "Dataloader's dataset should still be a dask array with chunks"
    # Assert the time coordinate is not a dask array
    assert dl.ds["time"].chunks is None

    for batch in dl:
        # Assert the inputs and targets returned by the batch are loaded in memory
        inputs, targets = batch.get_inputs_and_targets()
        assert inputs.initial_state["Wtot_MJ"].chunks is None, "Inputs initial_state should be loaded into memory (not a dask array)"
        assert inputs.inputs["B0"].chunks is None, "Inputs inputs should be loaded into memory (not a dask array)"
        assert isinstance(inputs.time, jnp.ndarray), "Time should be a numpy array (not a dask array)"
        assert targets["Wtot_MJ"].chunks is None, "Targets should be loaded into memory (not a dask array)"

def test_dask_time_indep_dataloader():
    # Test that the time-independent dataloader properly handles dask arrays and only brings them into memory as needed for JAX
    ds_path_nc =  os.path.join(DATA_DIR, "tcv", "scrambled_transport_sample.nc")
    ds = xr.open_dataset(ds_path_nc, chunks="auto")
    ds = ds.set_coords("time")

    # Ensure we start with a dask array
    assert ds["B0"].chunks is not None, "Dataset should be a dask array with chunks"

    dl = make_time_indep_dataloader(
        ds,
        time_coord="time",
        episode_coord="shot",
        input_vars=["B0"],
        target_vars=["Wtot_MJ"],
        convert_xr_to_jnp=False,  # Don't convert to jnp arrays, since we're testing the dataloader's handling of dask arrays
        batch_size=2, # Small batch size to ensure we're actually only loading one batch at a time
    )

    # Assert that the dataloader's dataset is still a dask array
    assert dl.ds["B0"].chunks is not None, "Dataloader's dataset should still be a dask array with chunks"
    # Assert the time coordinate is not a dask array
    assert dl.ds["time"].chunks is None, "Time-independent dataloader creation should have time coord consistent with time-dependent"

    for batch in dl:
        # Assert the inputs and targets returned by the batch are loaded in memory
        inputs, targets = batch.get_inputs_and_targets()
        assert inputs["B0"].chunks is None, "Inputs should be loaded into memory (not a dask array)"
        assert targets["Wtot_MJ"].chunks is None, "Targets should be loaded into memory (not a dask array)"

def test_dask_time_dep_training():
    # Test that we can train a simple time-dependent model from a dataset backed by dask arrays without running into issues with JAX
    # Here, invoking the transport predictor
    ds_path_nc = os.path.join(DATA_DIR, "tcv", "scrambled_transport_sample.nc")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_path_zarr = os.path.join(tmp_dir, "scrambled_transport_sample.zarr")
        ds = xr.open_dataset(ds_path_nc, chunks="auto")
        ds.to_zarr(ds_path_zarr)

        transport_predictor_config = TRANSPORT_PREDICTOR_BASE_CONFIG.model_copy(
            update={
                "dataloader_config": {
                    **TRANSPORT_PREDICTOR_BASE_CONFIG.dataloader_config,
                    "ds_path": ds_path_zarr,
                    "convert_xr_to_jnp": False,  # Don't convert to jnp arrays, since we're testing the dataloader's handling of dask arrays
                    "batch_size": 2, # Small batch size to ensure we're actually only loading a few samples at a time
                    "cheat_training": True, # Use same dataset for training and validation since the transport sample is too small to split, this is just to test that training runs without issues with dask arrays, not to get good performance
                },
                "max_epochs": 4, # Reduce epochs for testing purposes
                "epochs_per_val": 2,
            }
        )

        trainer, _, _, test_dl, _ = launch_train(transport_predictor_config.model_dump(), use_wandb=False)
        assert trainer is not None, "Trainer should be created successfully"
        assert test_dl is not None, "Test dataloader should be created successfully"

def test_dask_time_indep_training():
    # Test that we can train a simple time-independent model from a dataset backed by dask arrays without running into issues with JAX
    # Here, invoking the profile predictor submodule of the transport predictor
    ds_path_nc = os.path.join(DATA_DIR, "tcv", "scrambled_transport_sample.nc")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_path_zarr = os.path.join(tmp_dir, "scrambled_transport_sample.zarr")
        ds = xr.open_dataset(ds_path_nc, chunks="auto")
        ds.to_zarr(ds_path_zarr)

        profile_predictor_config = TrainConfig.load(TRANSPORT_PREDICTOR_BASE_CONFIG.model_init_config["submodules"]["profile_predictor"])
        profile_predictor_config = profile_predictor_config.model_copy(
            update={
                "dataloader_config": {
                    **profile_predictor_config.dataloader_config,
                    "ds_path": ds_path_zarr,
                    "convert_xr_to_jnp": False,  # Don't convert to jnp arrays, since we're testing the dataloader's handling of dask arrays
                    "batch_size": 2, # Small batch size to ensure we're actually only loading a few samples at a time
                    "cheat_training": True, # Use same dataset for training and validation since the transport sample is too small to split, this is just to test that training runs without issues with dask arrays, not to get good performance
                },
                "max_epochs": 4,
                "epochs_per_val": 2,
            }
        )

        trainer, _, _, test_dl, _ = launch_train(profile_predictor_config.model_dump(), use_wandb=False)
        assert trainer is not None, "Trainer should be created successfully"
        assert test_dl is not None, "Test dataloader should be created successfully"
