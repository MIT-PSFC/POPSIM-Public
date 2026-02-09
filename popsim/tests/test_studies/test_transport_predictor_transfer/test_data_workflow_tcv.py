import os
import pytest
import tempfile

import numpy as np
import xarray as xr

from popsim.data import get_path_to_ml_data_dump

from popsim.studies.transport_predictor_transfer.datasets.workflow import (
    DATASET_DT,
    EPISODE_DIM,
    TIME_COORD,
    TIME_DIM,
)

from popsim.studies.transport_predictor_transfer.datasets.tcv_data import (
    TCVDataWorkflow,
    TCV_SIGNAL_BOUNDS,
)

# Skip all tests in this module if ML data dump path is not available
pytestmark = pytest.mark.skipif(
    get_path_to_ml_data_dump() != "/usr/local/mfe/ml_data_dump/",
    reason="TCV data not available on this system",
    allow_module_level=True
)

@pytest.fixture(scope="session")
def cache_base():
    cache_base_dir = tempfile.gettempdir()
    return cache_base_dir

@pytest.fixture(scope="session")
def cached_raw_dataset_info(cache_base):
    """Create or reuse a cached TCV dataset in /tmp."""
    raw_ds_dir = os.path.join(cache_base, "raw")
    max_num_shots = 10
    min_shot_id = 53134

    workflow = TCVDataWorkflow(
        ds_name="tcv",
        is_target=True,
        max_num_shots=max_num_shots,
        min_shot_id=min_shot_id,
        raw_ds_dir=raw_ds_dir,
        processed_ds_dir=None,
        scratch_dir=None,
    )

    if not os.path.exists(workflow.raw_ds_path):
        workflow.build_dataset()

    return raw_ds_dir, max_num_shots, min_shot_id

@pytest.fixture(scope="session")
def cached_processed_dataset_info(cached_raw_dataset_info, cache_base):
    """Create or reuse a cached processed TCV dataset in /tmp."""
    raw_ds_dir, max_num_shots, min_shot_id = cached_raw_dataset_info
    processed_ds_dir = os.path.join(cache_base, "processed")

    workflow = TCVDataWorkflow(
        ds_name="tcv",
        is_target=True,
        max_num_shots=max_num_shots,
        min_shot_id=min_shot_id,
        raw_ds_dir=raw_ds_dir,
        processed_ds_dir=processed_ds_dir,
        scratch_dir=None,
    )

    if not os.path.exists(workflow.processed_ds_path):
        workflow.process_dataset()

    return raw_ds_dir, processed_ds_dir, max_num_shots, min_shot_id

def test_build_dataset():
    """Test building the raw TCV dataset with a few known shots.
    
    31650: No profile data available
    60000: Profile data, no NBI, has PradBulk
    61237: Profile data, with NBI, no PradBulk
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        for shot in [31650, 60000, 61237]:
            workflow = TCVDataWorkflow(
                ds_name="tcv",
                is_target=True,
                max_num_shots=1,
                min_shot_id=shot,
                raw_ds_dir=temp_dir,
                processed_ds_dir=None,
                scratch_dir=None,
            )
            workflow.build_dataset(extend_existing=True)
        
        # Ensure the dataset can be opened with consolidated metadata
        ds = xr.open_zarr(workflow.raw_ds_path, consolidated=True)

        # Check the timebase is something reasonable
        assert np.isclose(ds[TIME_COORD].isel({EPISODE_DIM: 0}).diff(TIME_DIM).mean(), DATASET_DT)

        # Ensure shot 31650 is not in the dataset since it's missing profiles
        assert 31650 not in ds[EPISODE_DIM].values

        # Ensure shot 60000 has NBI that's all nans and has valid PradBulk
        shot_data = ds.sel({EPISODE_DIM: 60000})
        assert shot_data["NBI"].isnull().all()
        assert shot_data["PradBulk"].notnull().any()

        # Ensure shot 61237 has valid NBI and a PradBulk that's all NaNs
        shot_data = ds.sel({EPISODE_DIM: 61237})
        assert shot_data["NBI"].notnull().any()
        assert shot_data["PradBulk"].isnull().all()


def test_process_dataset(cached_raw_dataset_info):
    """Test processing the raw TCV dataset and applying signal bounds."""
    raw_ds_dir, max_num_shots, min_shot_id = cached_raw_dataset_info
    with tempfile.TemporaryDirectory() as temp_dir:
        workflow = TCVDataWorkflow(
            ds_name="tcv",
            is_target=True,
            max_num_shots=max_num_shots,
            min_shot_id=min_shot_id,
            raw_ds_dir=raw_ds_dir,
            processed_ds_dir=temp_dir,
            scratch_dir=None,
        )
        workflow.process_dataset()

        ds = xr.open_zarr(workflow.processed_ds_path, consolidated=True).compute() # computing to more easily debug

        def _validate_signal(data, min_val, max_val):
            # Check for bounds, ignoring nans
            for shot in data[EPISODE_DIM].values:
                shot_data = data.sel({EPISODE_DIM: shot})
                if min_val is not None:
                    assert (shot_data.dropna(TIME_DIM) >= min_val).all()
                if max_val is not None:
                    assert (shot_data.dropna(TIME_DIM) <= max_val).all()

        # Ensure the dataset has no shots with invalid signal ranges
        for signal_name, config in TCV_SIGNAL_BOUNDS.items():
            min_val, max_val = config["bounds"]
            if "select" in config.keys():
                for dim, value in config["select"].items():
                    data = ds[signal_name].sel({dim: value})
                    _validate_signal(data, min_val, max_val)
            else:
                data = ds[signal_name]
                _validate_signal(data, min_val, max_val)


@pytest.mark.parametrize("extrapolate", ["chronological", "performance"])
def test_separate_test_set(extrapolate, cached_processed_dataset_info):
    """Test separating the processed TCV dataset into train and test sets using different extrapolation strategies."""
    raw_ds_dir, processed_ds_dir, max_num_shots, min_shot_id = cached_processed_dataset_info
    with tempfile.TemporaryDirectory() as temp_dir:
        workflow = TCVDataWorkflow(
            ds_name="tcv",
            is_target=True,
            max_num_shots=max_num_shots,
            min_shot_id=min_shot_id,
            raw_ds_dir=raw_ds_dir,
            processed_ds_dir=processed_ds_dir,
            scratch_dir=temp_dir,
            extrapolate=extrapolate,
        )
        workflow.separate_test_set()

        train_ds = xr.open_zarr(workflow.train_ds_path, consolidated=True)
        test_ds = xr.open_zarr(workflow.test_ds_path, consolidated=True)

        # Ensure no overlap in shots between train and test sets
        train_shots = set(train_ds[EPISODE_DIM].values)
        test_shots = set(test_ds[EPISODE_DIM].values)
        assert train_shots.isdisjoint(test_shots)

        if extrapolate == "chronological":
            # Ensure test set shots are all later than train set shots
            assert min(test_shots) > max(train_shots)
        elif extrapolate == "performance":
            # Ensure test set shots are all higher performance than train set shots
            train_performance = [train_ds.sel(shot=shot)["performance"].max().compute() for shot in train_shots]
            test_performance = [test_ds.sel(shot=shot)["performance"].max().compute() for shot in test_shots]
            assert min(test_performance) > max(train_performance)
