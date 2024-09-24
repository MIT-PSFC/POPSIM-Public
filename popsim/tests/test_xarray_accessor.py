import popsim.ml.xarray_accessor as xa
from popsim.tests.fixtures import cmod_test_dataset
import pytest

def test_assign_training_metadata(cmod_test_dataset):
    ds = cmod_test_dataset
    n_shots_keep = 50
    ds = ds.isel(shot=slice(0, n_shots_keep))

    data_var_names = list(ds.data_vars.keys())

    train_meta = xa.TrainingMetadata(
        sample_coord="shot",
        sample_dim="shot",
        param_vars=data_var_names[:2],
        target_vars=data_var_names[2:4],
        time_dep_metadata=xa.TrainingMetadata.TimeDepMetadata(
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