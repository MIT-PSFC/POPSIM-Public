import pytest
import xarray as xr

import popsim.ml
from popsim.tests.fixtures import (
    cmod_test_dataset,
    mast_thomson_test_dataset,
    oscillator_dataset,
    profile_predictor_latest_sparc,
    scrambled_multishot_liuqe_dataset,
    tcv_fbt_test_dataset,
)


@pytest.fixture(scope="session")
def all_datasets(
    cmod_test_dataset,
    mast_thomson_test_dataset,
    tcv_fbt_test_dataset,
    oscillator_dataset,
    scrambled_multishot_liuqe_dataset,
):
    """Define all of the datasets we will use for testing."""
    datasets = [
        cmod_test_dataset,
        mast_thomson_test_dataset,
        tcv_fbt_test_dataset,
        oscillator_dataset,
        scrambled_multishot_liuqe_dataset,
    ]

    return datasets

def test_generate_nan_report(all_datasets):
    """Test that generate_nan_report works on all dataset fixtures."""
    for ds in all_datasets:
        report, has_nans = ds.popsim_ml.generate_nan_report()

        # Report should be a string and has_nans should be a boolean
        assert isinstance(report, str)
        assert isinstance(has_nans, bool)

        # If NaNs are present, report should not be empty
        if has_nans:
            assert len(report) > 0