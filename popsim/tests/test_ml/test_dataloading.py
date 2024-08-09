from popsim.tests.test_ml.fixtures import cmod_test_dataset
from popsim.ml import make_dataloader
import xarray as xr
import numpy as np

def test_build_and_iterate_dataloader(cmod_test_dataset):
    ds = cmod_test_dataset
    n_shots_test = 100
    ds = ds.isel(shot=slice(0, n_shots_test)) # Downsampling for faster testing.
    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=["Wmhd"],
        other_vars=["p_rad"],
        segment_length=250,
        segment_overlap=25,
        batch_size=None,
    )

    # Check that only one iteration is run.
    for idx, batch in enumerate(dl):
        assert isinstance(batch, xr.Dataset)
        assert np.unique(batch["shot"]).size == n_shots_test # Check that all shots are present.
    assert idx == 0

    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=["Wmhd"],
        other_vars=["p_rad"],
        segment_length=250,
        segment_overlap=25,
        batch_size=64,
    )
    for idx, batch in enumerate(dl):
        assert isinstance(batch, xr.Dataset)
        assert dict(batch.sizes) == {'sample': 64, 'time_slice_input': 250}
    # Manually checked expected number of batches.
    assert idx == 10
