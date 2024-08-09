import typing

import numpy as np
import xarray as xr

DatasetOrDataArray = typing.Union[xr.Dataset, xr.DataArray]


def maybe_groupby_and_map(
    ds: DatasetOrDataArray, dim: str, func: typing.Callable[[DatasetOrDataArray], DatasetOrDataArray]
) -> DatasetOrDataArray:
    """If the dataset(array) has the specified dimension, group by that dimension and apply the function to each group. Otherwise, apply the function to the entire dataset(array).

    Args:
        ds (DatasetOrDataArray): dataset(array) to be processed.
        dim (str): name of the dimension to group by.
        func (typing.Callable[[DatasetOrDataArray], DatasetOrDataArray]): function to be applied to each group.

    Returns:
        DatasetOrDataArray: dataset(array) with the function applied to each group.
    """

    if dim in ds.dims:
        return ds.groupby(dim).map(func)
    else:
        return func(ds)


def mask_to_largest_group_mask(mask: xr.DataArray, episode_dim: str, time_dim: str) -> xr.DataArray:
    """Given a boolean mask, generate a new mask that only includes the largest
    group of contiguous True/1 values for each shot. In the case of a tie, the first group is selected.

    Args:
        mask (xr.DataArray): boolean mask to be processed. Int arrays of 0 and 1 are also accepted.
        episode_dim (str): name of the episode dimension (e.g. "shot").
        time_dim (str): name of the time dimension (e.g. "time").

    Returns:
        xr.DataArray: mask with only the largest group of contiguous True/1 values for each shot.
    """

    def _mask_to_largest_group_mask(mask: xr.DataArray) -> xr.DataArray:
        # Check that the mask is int or bool. If it is bool, convert to int.
        if mask.dtype == "bool":
            mask = mask.astype(int)
        elif mask.dtype == "int":
            assert np.all(np.logical_or(mask == 0, mask == 1))
        else:
            raise ValueError(f"Expected mask to be of type int or bool, but got {mask.dtype}")

        # Remove extraneous dimensions.
        mask = mask.squeeze()

        # If the mask is all False, there are no groups to consider.
        if (mask == 0).all():
            return mask

        # At every point where the mask changes, increment a cumulative sum.
        # This results in a unique number for each continguous group of Trues and Falses.
        cumulative_number_of_mask_flips = (mask != mask.shift({time_dim: 1})).cumsum()

        # Label all points where the mask is True with -1. Once that is done, what is left
        # are the group numbers for each contiguous group of True values.
        groups = cumulative_number_of_mask_flips.where(mask, other=-1)

        # Compute the sizes of the groups and find the largest group.
        group_sizes = groups.to_pandas().value_counts().drop(-1, errors="ignore")
        largest_group = group_sizes.idxmax()
        return groups == largest_group

    return maybe_groupby_and_map(mask, episode_dim, _mask_to_largest_group_mask)


def shift_time_to_not_nan(
    ds: xr.Dataset, episode_dim: str, time_var: str, how: str = "any", subset: typing.Optional[typing.Iterable[typing.Hashable]] = None
) -> xr.Dataset:
    """For each episode in a dataset, shift the time dimension so that the first time slice is not NaN. Each episode is end-padded with NaNs. To specify that only a subset of the variables should be considered when determining the first non-NaN time slice, pass a list of variable names to the `subset` argument.

    Args:
        ds (xr.Dataset): dataset to be processed.
        episode_dim (str): name of the episode dimension (e.g. "shot").
        time_var (str): name of the time variable (e.g. "time").
        how (str, optional): Either "any" or "ally". Forwarded to xr.Dataset.dropna . Defaults to "any".
        subset (typing.Optional[typing.Iterable[typing.Hashable]], optional): Forwarded to xr.Dataset.dropna . Defaults to None.

    Returns:
        xr.Dataset: dataset with the time dimension shifted so that the first time slice is not NaN.
    """

    def _shift_time_to_not_nan(group):
        # Remove extraneous dimensions.
        group = group.squeeze()

        assert len(group[time_var].dims) == 1, "Unexpected number of dimensions for time variable."
        time_dim = group[time_var].dims[0]

        # Drop NaNs across the 'time_slice' dimension.
        cleaned_group = group.dropna(time_dim, how=how, subset=subset)

        # If there are no time slices left, just return the cleaned group which is empty.
        if cleaned_group[time_dim].size == 0:
            return cleaned_group

        # Find the first time slice remaining in the cleaned group.
        # Find its index in the original group and shift the time dimension by that amount.
        first_cleaned_slice = cleaned_group[time_var].isel({time_dim: 0})
        time_index = np.where(group[time_var] == first_cleaned_slice)[0][0]
        n_shift = time_index

        # We want to shift the time dimension by n_shift, but .shift() does not shift
        # the coordinates, which we want to do. So we first use reset_coords() to convert
        # the coordinates to normal variables, apply the shift, then set the coordinates back.
        group_coords = list(group.coords)
        group = group.reset_coords()
        shifted_group = group.shift({time_dim: -n_shift})
        shifted_group = shifted_group.set_coords(group_coords)
        return shifted_group

    return maybe_groupby_and_map(ds, episode_dim, _shift_time_to_not_nan)
