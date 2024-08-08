import typing

import numpy as np
import xarray as xr
import xbatcher


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

    return mask.groupby(episode_dim).map(lambda shot_mask: _mask_to_largest_group_mask(shot_mask))


def shift_time_to_not_nan(
    ds: xr.Dataset, episode_dim: str, time_var: str, how: str = "any", subset: typing.Optional[typing.Iterable[typing.Hashable]] = None
) -> xr.Dataset:
    """For each shot in a dataset, shift the time dimension so that the first time slice is not NaN. Each shot is end-padded with NaNs. To specify that only a subset of the variables should be considered when determining the first non-NaN time slice, pass a list of variable names to the `subset` argument.

    Args:
        ds (xr.Dataset): dataset to be processed.
        episode_dim (str): name of the episode dimension (e.g. "shot").
        time_var (str): name of the time variable (e.g. "time").
        how (str, optional): Either "any" or "ally". Forwarded to xr.Dataset.dropna . Defaults to "any".
        subset (typing.Optional[typing.Iterable[typing.Hashable]], optional): Forwarded to xr.Dataset.dropna . Defaults to None.

    Returns:
        xr.Dataset: dataset with the time dimension shifted so that the first time slice is not NaN.
    """

    def dropna_preserve_size(group):
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
        n_shift = -time_index
        return group.shift({time_dim: n_shift})

    return ds.groupby(episode_dim).map(dropna_preserve_size)


def make_sample_ds(
    ds: xr.Dataset,
    time_var: str,
    episode_var: str,
    state_init_vars: list[str],
    param_vars: list[str],
    segment_length: int,
    segment_overlap: int,
) -> xr.Dataset:
    """Given a dataset,

    Args:
        ds (xr.Dataset): input dataset.
        time_name (str): Name of the time variable.
        episode_name (str): Name of the episode variable.
        state_init_vars (list[str]): List of variables required to initialize the state of the model.
        param_vars (list[str]): List of variables that are time-varying parameters input to the model.
        batch_size (int): Size of training batches.
        segment_length (int): Number of time steps used in each training segment.
        segment_overlap (int): Number of time steps that each segment overlaps with the previous segment.

    Returns:
        xr.Dataset: Dataset with dimensions (sample, time, other_dims).
    """

    def _get_and_check_episode_and_time_dims(ds: xr.Dataset, episode_var_name: str, time_var_name: str):
        # Extract the dimension of the episode variable and make sure it is one-dimensional.
        episode_var = ds[episode_var_name]
        episode_var_dims = list(episode_var.sizes.keys())
        assert len(episode_var_dims) == 1, f"Expected one dimension for episode var {episode_var}"
        episode_var_dim = episode_var_dims[0]

        # Extract the dimension of the time variable.
        time_var = ds[time_var_name]
        time_var_dims = set(time_var.sizes.keys())

        assert episode_var_name in time_var_dims, f"{episode_var} not in {time_var_dims}"
        time_var_dims_minus_episode = [d for d in time_var_dims if d != episode_var_dim]

        assert (
            len(time_var_dims_minus_episode) == 1
        ), f"For time var {time_var}, expected one dimension besides the episode dimension, got {time_var_dims_minus_episode}"
        time_var_dim = time_var_dims_minus_episode[0]
        return episode_var_dim, time_var_dim

    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_var, time_var)

    ds = shift_time_to_not_nan(ds, episode_dim=episode_var_dim, time_var=time_var, how="any", subset=state_init_vars)

    # Construct the input_dims dictionary to define the input dimension the model will see.
    # We want the model to see a fixed number of time steps and data from a single episode.
    # However, we want to keep all other dimensions the same.
    non_episode_time_dims = {k: v for k, v in ds.sizes.items() if k not in [episode_var_dim, time_var_dim]}
    input_dims = {time_var_dim: segment_length} | non_episode_time_dims

    # Use a first call to Xbatcher to segment the episodes to the desired length and create a dataset with dimensions
    # (sample, time, other_dims).
    sample_ds = next(
        iter(xbatcher.BatchGenerator(ds, input_dims=input_dims, input_overlap={time_var_dim: segment_overlap}, concat_input_dims=True))
    )

    # Drop samples where the data is all NaN.
    sample_ds = sample_ds.dropna("sample", how="all", subset=state_init_vars + param_vars)

    return sample_ds
