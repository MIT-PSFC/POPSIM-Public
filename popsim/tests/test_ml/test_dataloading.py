import pytest
from popsim.tests.fixtures import cmod_test_dataset
from popsim.ml.dataloading import make_dataloader, make_time_indep_dataloader
import xarray as xr
import numpy as np

@pytest.fixture
def reduced_cmod_test_dataset(cmod_test_dataset):
    # Downsample to 100 shots for faster testing.
    ds = cmod_test_dataset
    n_shots_test = 100
    ds = ds.isel(shot=slice(0, n_shots_test))
    state_init_vars = ["Wmhd"]
    param_vars = ["ip", "n_e"]
    targ_vars = ["Wmhd"]
    return ds, state_init_vars, param_vars, targ_vars

def test_make_dataloader(reduced_cmod_test_dataset):
    ds, state_init_vars, param_vars, targ_vars = reduced_cmod_test_dataset

    #
    # Test full-batch dataloading for segmented data.
    #
    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=state_init_vars,
        param_vars=param_vars,
        targ_vars=targ_vars,
        segment_length=250,
        segment_overlap=25,
        batch_size=None,
    )
    # Since batch_size is None, the dataloader should have only one batch.
    assert len(dl) == 1

    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=state_init_vars,
        param_vars=param_vars,
        targ_vars=targ_vars,
        segment_length=250,
        segment_overlap=25,
        batch_size=64,
    )
    for idx, batch in enumerate(dl):
        batch_input, batch_target = batch
        assert isinstance(batch_input, xr.Dataset)
        assert isinstance(batch_target, xr.Dataset)
        assert dict(batch_input.sizes) == {'sample': 64, 'time_slice_input': 250}
        assert dict(batch_target.sizes) == {'sample': 64, 'time_slice_input': 250}
    # Check against the manually determined number of batches.
    assert idx == 10

def test_make_time_indep_dataloader(reduced_cmod_test_dataset):
    ds, _, _, _ = reduced_cmod_test_dataset
    #
    dl = make_time_indep_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        input_vars=["Wmhd", "p_rad", "ip", "n_e"],
        targ_vars=["Wmhd"],
        batch_size=64,
    )
    for _, batch in enumerate(dl):
        batch_input, batch_target = batch
        assert isinstance(batch_input, xr.Dataset)
        assert isinstance(batch_target, xr.Dataset)
        assert dict(batch_input.sizes) == {'sample': 64}
        assert dict(batch_target.sizes) == {'sample': 64}

def test_collate(reduced_cmod_test_dataset):
    ds, state_init_vars, param_vars, targ_vars = reduced_cmod_test_dataset

    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=state_init_vars,
        param_vars=param_vars,
        targ_vars=targ_vars,
        segment_length=250,
        segment_overlap=0,
        batch_size=None,
    )

    inputs, targets = next(iter(dl))

    def ds_to_numpy(ds: xr.Dataset):
        return {var: ds[var].to_numpy() for var in ds.data_vars}

    # Grab the first time slice of every sample.
    state_init = ds_to_numpy(inputs[state_init_vars].isel({dl.dataloader.dataset.time_dim: 0}))
    params = ds_to_numpy(inputs[param_vars])
    time = inputs[dl.dataloader.dataset.time_var].values