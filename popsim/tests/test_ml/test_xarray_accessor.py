from popsim.ml._types import TrainingMetadata
from popsim.ml import DEFAULT_SAMPLE_DIM
from popsim.tests.fixtures import cmod_test_dataset, reduced_cmod_test_dataset
import pytest

def test_assign_training_metadata(cmod_test_dataset):
    ds = cmod_test_dataset
    n_shots_keep = 50
    ds = ds.isel(shot=slice(0, n_shots_keep))

    data_var_names = list(ds.data_vars.keys())

    train_meta = TrainingMetadata(
        sample_coord="shot",
        sample_dim="shot",
        input_vars=data_var_names[:2],
        target_vars=data_var_names[2:4],
        episode_coord="shot",
        episode_dim="shot",
        time_dep_metadata=TrainingMetadata.TimeDepMetadata(
            state_init_vars=data_var_names[4:],
            time_coord="time",
            time_dim="time",
        )
    )

    ds.popsim_ml.training_metadata = train_meta
    assert ds.popsim_ml.training_metadata == train_meta
    assert ds.popsim_ml.n_samples == n_shots_keep
    assert ds.popsim_ml.sample_dim == "shot"

    # Test that if a 2D sample coordinate is assigned, it raises a ValueError
    train_meta.sample_coord = data_var_names[0] # Data vars are 2D
    with pytest.raises(ValueError):
        ds.popsim_ml.training_metadata = train_meta

@pytest.mark.parametrize("split_fracs", [[0.7, 0.15, 0.15], [0.6, 0.4]])
def test_make_standard_dataloader(reduced_cmod_test_dataset, split_fracs):
    ds, state_init_vars, input_vars, target_vars = reduced_cmod_test_dataset
    
    # Check the time independent case.
    dls = ds.popsim_ml.make_dataloaders(
        time_coord="time",
        episode_coord="shot",
        input_vars=["p_rad", "ip", "n_e"],
        target_vars=["Wmhd"],
        split_fracs=split_fracs,
        key=42,
        batch_size=1024,
    )
    assert len(dls) == len(split_fracs)
    assert dls[0].dl.shuffle == True
    assert dls[1].dl.shuffle == False
    if len(dls) == 3:
        assert dls[2].dl.shuffle == False
    batch = next(iter(dls[0].dl))
    assert batch.ds.sizes[DEFAULT_SAMPLE_DIM] == 1024
    
    # Check the time dependent case.
    dls = ds.popsim_ml.make_dataloaders(
        time_coord="time",
        episode_coord="shot",
        input_vars=input_vars,
        target_vars=target_vars,
        split_fracs=split_fracs,
        key=42,
        state_init_vars=state_init_vars,
        batch_size=1024,
        segment_length=50,
    )
    assert len(dls) == len(split_fracs)
    assert dls[0].dl.shuffle == True
    assert dls[1].dl.shuffle == False
    if len(dls) == 3:
        assert dls[2].dl.shuffle == False
    batch = next(iter(dls[0]))
    assert batch.ds.sizes['time_slice_input'] == 50
    assert batch.ds.sizes[DEFAULT_SAMPLE_DIM] == 1024