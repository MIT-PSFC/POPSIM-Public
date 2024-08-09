from popsim.tests.test_ml.fixtures import cmod_test_dataset
from popsim.ml.dataloading import make_dataloader

def test_build_and_iterate_dataloader(cmod_test_dataset):
    ds = cmod_test_dataset
    ds = ds.isel(shot=slice(0, 50))
    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=["Wmhd"],
        param_vars=["p_rad"],
        segment_length=250,
        segment_overlap=25,
        batch_size=None,
    )
    # Possible bug in jax_dataloader.
    # https://github.com/BirkhoffG/jax-dataloader/issues/33
    assert len(dl) == 1

    dl = make_dataloader(
        ds=ds,
        time_var="time",
        episode_var="shot",
        state_init_vars=["Wmhd"],
        param_vars=["p_rad"],
        segment_length=250,
        segment_overlap=25,
        batch_size=128,
    )
    for batch in dl:
        pass