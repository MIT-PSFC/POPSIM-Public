import pytest
import xarray as xr
import numpy as np

from popsim.ml.preprocess_utils import mask_to_largest_group_mask, shift_time_to_not_nan

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
                "var1": (("shot", "time"), [[1, 2, 3], [4, 5, 6]]),
                "var2": (("shot", "time"), [[7, 8, 9], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[1, 2, np.nan], [6, np.nan, np.nan]]),
                "var2": (("shot", "time"), [[8, 9, np.nan], [12, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[2, np.nan, np.nan], [5, np.nan, np.nan]]),
                "var2": (("shot", "time"), [[9, np.nan, np.nan], [11, 12, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[1, 2, np.nan], [4, np.nan, 6]]),
                "var2": (("shot", "time"), [[8, 9, np.nan], [np.nan, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[2, np.nan, np.nan], [np.nan, np.nan, 6]]),
                "var2": (("shot", "time"), [[9, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [5, np.nan, np.nan]]),
                "var2": (("shot", "time"), [[9, np.nan, np.nan], [11, 12, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[1, 2, np.nan], [4, np.nan, 6]]),
                "var2": (("shot", "time"), [[8, 9, np.nan], [np.nan, np.nan, np.nan]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
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
                "var1": (("shot", "time"), [[np.nan, np.nan, np.nan], [4, 5, 6]]),
                "var2": (("shot", "time"), [[np.nan, np.nan, np.nan], [10, 11, 12]])
            },
            coords={"time": [0.5, 1.5, 2.0], "shot": [0, 1]}
        )
    ),
])
def test_shift_time_to_not_nan(ds, how, subset, expected):
    result = shift_time_to_not_nan(ds, episode_dim="shot", time_var="time", how=how, subset=subset)
    xr.testing.assert_identical(result, expected)