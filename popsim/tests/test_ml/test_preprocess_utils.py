import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from popsim import PACKAGE_ROOT
from popsim.ml.dataloading import make_time_dep_dataloader
from popsim.ml.preprocess_utils import expand_time_dim, mask_to_largest_group_mask, shift_time_to_not_nan, trim_time_to_not_nan
from popsim.tests.fixtures import cmod_test_dataset, mast_thomson_test_dataset


@pytest.fixture(scope="session")
def cmod_data_with_energy_mask(cmod_test_dataset):
    ds = cmod_test_dataset
    energy_bounds = (1e4, 2.5e5)
    ds["values_in_bounds"] = (energy_bounds[0] < ds["Wmhd"]) & (ds["Wmhd"] < energy_bounds[1])
    return ds


@pytest.mark.parametrize("mask, expected", [
    # Single Episode, Single Group
    (
        xr.DataArray([[0, 1, 1, 0, 1]], dims=["episode", "time"]),
        xr.DataArray([[0, 1, 1, 0, 0]], dims=["episode", "time"])
    ),
    # Single Episode, Multiple Groups
    (
        xr.DataArray([[1, 1, 0, 1, 1, 1, 0, 1]], dims=["episode", "time"]),
        xr.DataArray([[0, 0, 0, 1, 1, 1, 0, 0]], dims=["episode", "time"])
    ),
    # Multiple Episodes
    (
        xr.DataArray([[1, 1, 0, 1], [0, 1, 1, 1]], dims=["episode", "time"]),
        xr.DataArray([[1, 1, 0, 0], [0, 1, 1, 1]], dims=["episode", "time"])
    ),
    # All False
    (
        xr.DataArray([[0, 0, 0, 0]], dims=["episode", "time"]),
        xr.DataArray([[0, 0, 0, 0]], dims=["episode", "time"])
    ),
    # All True
    (
        xr.DataArray([[1, 1, 1, 1]], dims=["episode", "time"]),
        xr.DataArray([[1, 1, 1, 1]], dims=["episode", "time"])
    ),
    # Non-Contiguous Dimension
    (
        xr.DataArray([[1, 0, 1, 0, 1]], dims=["episode", "time"]),
        xr.DataArray([[1, 0, 0, 0, 0]], dims=["episode", "time"])
    )
])
def test_mask_to_largest_group_mask(mask, expected):
    result = mask_to_largest_group_mask(mask, episode_dim="episode", time_dim="time")
    xr.testing.assert_equal(result, expected)

def test_mask_to_largest_group_mask_invalid_dtype():
    mask = xr.DataArray(
        [[0.5, 1.0, 0.0, 1.0]],
        dims=["episode", "time"]
    )
    with pytest.raises(ValueError, match="Expected mask to be of type int or bool"):
        mask_to_largest_group_mask(mask, episode_dim="episode", time_dim="time")

@pytest.mark.parametrize("multishot", [True, False])
def test_test_mask_to_largest_group_cmod(cmod_data_with_energy_mask, multishot):
    ds = cmod_data_with_energy_mask

    # Test case and expected result was derived manually via Jupyter Notebook.
    test_shot = 1140723019
    expected_times_true = (0.33699992, 1.61799992)

    def get_times_true(_ds):
        _ds["values_in_bounds_largest_group"] = mask_to_largest_group_mask(_ds["values_in_bounds"], episode_dim="shot", time_dim="time_slice")
        times_true = _ds["time"].where(_ds["values_in_bounds_largest_group"])
        return times_true

    if multishot:
        # Run the function with the entire dataset.
        times_true = get_times_true(ds)
        times_true = times_true.sel(shot=test_shot)
    else:
        # Run the function with a single shot.
        times_true = get_times_true(ds.sel(shot=test_shot))

    # Check that the first and last times that are True are as expected.
    # Also check that all times in between the two times are true.
    assert np.isclose(times_true.min().values, expected_times_true[0])
    assert np.isclose(times_true.max().values, expected_times_true[1])
    assert times_true.dropna("time_slice").all()


@pytest.mark.parametrize("ds, how, subset, expected", [
    # how=Any: No NaN
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[1, 2, 3], [4, 5, 6]]),
                "var2": (("shot", "time"), [[7, 8, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[1, 2, 3], [4, 5, 6]]),
                "var2": (("shot", "time_slice"), [[7, 8, 9], [10, 11, 12]])
            },
            coords = {
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            },
        )
    ),
    # how=Any: NaNs at Start
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, 1, 2], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[1, 2, np.nan], [6, np.nan, np.nan]]),
                "var2": (("shot", "time_slice"), [[8, 9, np.nan], [12, np.nan, np.nan]])
            },
            coords = {
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[1.5, 2.0, np.nan], [2.0, np.nan, np.nan]]),
            },
        )
    ),
    # how=Any: NaNs at Start and Middle
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, 2], [np.nan, 5, np.nan]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [np.nan, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[2, np.nan, np.nan], [5, np.nan, np.nan]]),
                "var2": (("shot", "time_slice"), [[9, np.nan, np.nan], [11, 12, np.nan]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[2.0, np.nan, np.nan], [1.5, 2.0, np.nan]]),
            },
        )
    ),
    # how=Any: Only use a subset of variables for determining the first non-NaN time slice
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, 1, 2], [4, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [np.nan, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        ["var1"],
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[1, 2, np.nan], [4, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[8, 9, np.nan], [np.nan, np.nan, np.nan]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[1.5, 2.0, np.nan], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=Any: All NaNs in a Shot
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=All: NaNs at Start
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, 2], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[2, np.nan, np.nan], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[9, np.nan, np.nan], [10, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[2.0, np.nan, np.nan], [0.5, 1.5, 2.0]]),
            },
        )
    ),
    # how=All: NaNs at Start and Middle
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [np.nan, 5, np.nan]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, 9], [np.nan, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [5, np.nan, np.nan]]),
                "var2": (("shot", "time_slice"), [[9, np.nan, np.nan], [11, 12, np.nan]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[2.0, np.nan, np.nan], [1.5, 2.0, np.nan]]),
            }
        )
    ),
    # how=All: Only use a subset of variables for determining the first non-NaN time slice
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, 1, 2], [4, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [np.nan, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        ["var1"],
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[1, 2, np.nan], [4, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[8, 9, np.nan], [np.nan, np.nan, np.nan]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[1.5, 2.0, np.nan], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=All: All NaNs in a Shot
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (("shot", "time_slice"), [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]])
            }
        )
    ),
])
def test_shift_time_to_not_nan(ds, how, subset, expected):

    episode_dim = "shot"
    time_coord = "time"

    ds, time_dim = expand_time_dim(ds, episode_dim, time_coord, f"{time_coord}_slice")

    result = shift_time_to_not_nan(ds, episode_dim, time_coord, time_dim, how=how, subset=subset)
    xr.testing.assert_identical(result, expected)

@pytest.mark.parametrize("multishot", [True, False])
def test_shift_time_to_not_nan_cmod(cmod_data_with_energy_mask, multishot):
    ds = cmod_data_with_energy_mask
    test_shot = 1120104005

    # Expected first and last time and Wmhd values after shifting the time dimension.
    expected_times = (0.347, 1.774)
    expected_wmhds = (10135.40933778, 10013.27040793)
    ds = ds.where(ds["values_in_bounds"], drop=True)

    episode_dim = "shot"
    time_coord = "time"
    time_dim = "time_slice"
    if episode_dim not in ds[time_coord].dims:
        ds, time_dim = expand_time_dim(ds, episode_dim, time_coord, time_dim)
    if multishot:
        result = shift_time_to_not_nan(ds, episode_dim, time_coord, time_dim, how="any", subset=["Wmhd"])
        result = result.sel(shot=test_shot)
    else:
        result = shift_time_to_not_nan(ds.sel(shot=test_shot), episode_dim, time_coord, time_dim, how="any", subset=["Wmhd"])
    
    result = result.dropna('time_slice', how='all')
    # Check that the first and last times and Wmhd values are as expected.
    assert np.isclose(result["time"].min().values, expected_times[0])
    assert np.isclose(result["time"].max().values, expected_times[1])
    assert np.isclose(result["Wmhd"].isel(time_slice=0), expected_wmhds[0])
    assert np.isclose(result["Wmhd"].isel(time_slice=-1), expected_wmhds[1])

@pytest.mark.parametrize("ds, how, subset, expected", [
    # how=Any: No NaN
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[1, 2, 3], [4, 5, 6]]),
                "var2": (("shot", "time"), [[7, 8, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[1, 2, 3], [4, 5, 6]]),
                "var2": (("shot", "time_slice"), [[7, 8, 9], [10, 11, 12]])
            },
            coords = {
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            },
        )
    ),
    # how=Any: NaNs at Start
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, 1, 2], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, 1, 2], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, 8, 9], [10, 11, 12]])
            },
            coords = {
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            },
        )
    ),
    # how=Any: NaNs at Start and Middle
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, 2], [np.nan, 5, np.nan]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [np.nan, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, 2], [np.nan, 5, np.nan]]),
                "var2": (("shot", "time_slice"), [[np.nan, 8, 9], [np.nan, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, np.nan]]),
            },
        )
    ),
    # how=Any: Only use a subset of variables for determining the first non-NaN time slice
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, 1, 2], [4, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [np.nan, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        ["var1"],
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, 1, 2], [4, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, 8, 9], [np.nan, np.nan, np.nan]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=Any: All NaNs in a Shot
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "any",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[np.nan, np.nan, np.nan], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=All: NaNs at Start
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, 2], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, 2], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, np.nan, 9], [10, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            },
        )
    ),
    # how=All: NaNs at Start and Middle
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [np.nan, 5, np.nan]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, 9], [np.nan, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [np.nan, 5, np.nan]]),
                "var2": (("shot", "time_slice"), [[np.nan, np.nan, 9], [np.nan, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=All: Only use a subset of variables for determining the first non-NaN time slice
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, 1, 2], [4, np.nan, 6]]),
                "var2": (("shot", "time"), [[np.nan, 8, 9], [np.nan, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        ["var1"],
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, 1, 2], [4, np.nan, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, 8, 9], [np.nan, np.nan, np.nan]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (["shot", "time_slice"], [[0.5, 1.5, 2.0], [0.5, 1.5, 2.0]]),
            }
        )
    ),
    # how=All: All NaNs in a Shot
    (
        xr.Dataset(
            {
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        ),
        "all",
        None,
        xr.Dataset(
            {
                "var1": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={
                "shot": [0, 1],
                "time_slice": [0, 1, 2],
                "time": (("shot", "time_slice"), [[np.nan, np.nan, np.nan], [0.5, 1.5, 2.0]])
            }
        )
    ),
])
def test_trim_time_to_not_nan(ds, how, subset, expected):

    episode_dim = "shot"
    time_coord = "time"

    ds, time_dim = expand_time_dim(ds, episode_dim, time_coord, f"{time_coord}_slice")

    result = trim_time_to_not_nan(ds, episode_dim, time_coord, time_dim, how=how, subset=subset)
    xr.testing.assert_identical(result, expected)