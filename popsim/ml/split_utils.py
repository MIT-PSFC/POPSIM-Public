import math
from collections.abc import Sequence
from itertools import accumulate
from typing import Optional, Union

import jax
import numpy as np
import xarray as xr


def fracs_to_lengths(n_data: int, fracs: Sequence[float]) -> list[int]:
    """Convert a sequence of fractions to a list of lengths. Intended as a helper function for splitting datasets by fractions.

    Args:
        n_data (int): number of data points.
        fracs (Sequence[float]): sequence of fractions.

    Returns:
        list[int]: list of lengths that should add up to n_data.
    """
    if not isinstance(n_data, int) or n_data <= 0:
        raise ValueError("n_data must be a positive integer.")

    if round(sum(fracs), 2) != 1.0:
        raise ValueError("Sum of fractions in fracs must add up to 1.0.")

    # Calculate segment lengths and round them
    segment_lengths = [round(frac * n_data) for frac in fracs]

    # Adjust the segment lengths to ensure they sum up to n_data
    while sum(segment_lengths) != n_data:
        if sum(segment_lengths) > n_data:
            segment_lengths[segment_lengths.index(max(segment_lengths))] -= 1
        else:
            segment_lengths[segment_lengths.index(min(segment_lengths))] += 1
    if not sum(segment_lengths) == n_data:
        raise ValueError("Sum of segment lengths does not equal n_data. This is a bug, please report it.")
    return segment_lengths


def random_split(n_data: int, lengths_or_fracs: Sequence[Union[int, float]], seed: int) -> list[jax.Array]:
    """Generate indicies to split a dataset into non-overlapping new datasets.

    Args:
        n_data (int): size of the dataset to be split.
        lengths_or_fracs (Sequence[Union[int, float]]): lengths or fractions of splits to be produced.
        seed (int): seed for psuedo-random number generation.

    Returns:
        List[jax.Array]: list of indicies to split the dataset.
    """
    if math.isclose(sum(lengths_or_fracs), 1) and sum(lengths_or_fracs) <= 1:
        lengths = fracs_to_lengths(n_data, lengths_or_fracs)
    else:
        if sum(lengths_or_fracs) != n_data:
            raise ValueError("Sum of input lengths does not equal n_data!")
        lengths = lengths_or_fracs

    # Cannot verify that dataset is Sized
    if sum(lengths) != n_data:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = jax.random.permutation(
        jax.random.PRNGKey(seed),
        n_data,
    )
    return [indices[offset - length : offset] for offset, length in zip(accumulate(lengths), lengths)]


def split_dataset_by_fracs(
    ds: xr.Dataset,
    fracs: Sequence[float],
    dim: str,
    seed: int,
    sortby: Optional[str] = None,
) -> Sequence[xr.Dataset]:
    """Split a dataset into disjoint datasets along a dimension. The most common use case is for splitting a dataset along a "sample" dimension into training, validation, and test sets.

    Args:
        ds (xr.Dataset): dataset to be split.
        fracs (Sequence[float]): fractions of splits to be produced.
        dim (str): dimension to split the dataset along.
        seed (int): seed for pseudo-random number generation.
        sortby (str, optional): If provided, the dataset will be sorted in ascending order according to maximum value within each dimension. e.g. if sortby='betap' low-performance shots will come before high-performance shots.

    Returns:
        Sequence[xr.Dataset]: sequence of datasets.
    """

    n_data = ds.sizes[dim]
    lengths = fracs_to_lengths(n_data, fracs)

    if sortby:
        # Sort dataset by maximum value of sortby variable along the specified dimension
        sort_values = ds[sortby].max(dim=[d for d in ds[sortby].dims if d != dim], skipna=True)
        sorted_indices = np.argsort(sort_values.values)
        ds = ds.isel({dim: sorted_indices})

        split_idxs = [np.arange(offset - length, offset) for offset, length in zip(accumulate(lengths), lengths)]
        dataset_splits = [ds.isel({dim: np.asarray(idxs)}) for idxs in split_idxs]
    else:
        dataset_splits = [ds.isel({dim: np.asarray(idxs)}) for idxs in random_split(n_data, lengths, seed)]

    return dataset_splits


def split_dataset_by_vals(ds: xr.Dataset, vals: Sequence[Sequence], dim: str) -> tuple[xr.Dataset, Sequence[xr.Dataset]]:
    """Split a dataset into multiple datasets along a dimension by specifying the values of the dimension to include in each split.

    Args:
        ds (xr.Dataset): dataset to be split.
        vals (Sequence[Sequence]): values of dim to include in each split.
        dim (str): dimension to split the dataset along.

    Returns:
        dataset_splits: Sequence[xr.Dataset]: sequence of datasets with the vals of dim specified.
        reduced_dataset: xr.Dataset: dataset with the vals of dim specified removed.
    """

    covered_vals = np.concatenate(vals)
    missing_values = [val for val in covered_vals if val not in ds[dim]]
    if missing_values:
        raise ValueError(f"The following values in splits are not in ds[{dim}]: {missing_values}")

    dataset_splits = [ds.sel({dim: np.asarray(val)}) for val in vals]
    reduced_dataset = ds.drop_sel({dim: covered_vals})
    return dataset_splits, reduced_dataset
