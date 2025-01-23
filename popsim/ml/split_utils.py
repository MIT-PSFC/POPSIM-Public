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


def split_dataset_along_dim(
    ds: xr.Dataset,
    fracs: Sequence[float],
    dim: str,
    seed: int,
    ordered: Optional[bool] = False,
    pre_split: Optional[Sequence[Sequence]] = None,
) -> Sequence[xr.Dataset]:
    """Split a dataset into disjoint datasets along a dimension. The most common use case is for splitting a dataset along a "sample" dimension into training, validation, and test sets.

    Args:
        ds (xr.Dataset): dataset to be split.
        fracs (Sequence[float]): fractions of splits to be produced.
        dim (str): dimension to split the dataset along.
        seed (int): seed for psuedo-random number generation.
        ordered (bool, optional): whether the datasets should be sorted chronologically, with the the first dataset containing the earliest data. Defaults to False.
        pre_split (Sequence[Sequence], optional): Values of dim to include in each split. Defaults to None. Throws an error if the lengths of the pre_split and fracs are not equal, if there are too many values for the split fraction, or if the pre_split values are not a subset of the values in ds[dim].

    Returns:
        Sequence[xr.Dataset]: sequence of datasets.
    """

    n_data = ds.sizes[dim]
    lengths = fracs_to_lengths(n_data, fracs)

    # Take out the pre_split values beforehand and adjust lengths accordingly
    if pre_split is not None:
        if len(pre_split) != len(fracs):
            raise ValueError(f"The lengths of the pre_split ({len(pre_split)}) and fracs ({len(fracs)}) must be equal.")

        covered_vals = np.concatenate(pre_split)
        missing_values = [val for val in covered_vals if val not in ds[dim]]
        if missing_values:
            raise ValueError(f"The following values in pre_split are not in ds[dim]: {missing_values}")

        original_lengths = lengths
        lengths = [length - len(vals) for length, vals in zip(lengths, pre_split)]
        if any(length < 0 for length in lengths):
            raise ValueError(
                f"Too many pre_split values for the split fractions. Allocated {original_lengths} from {fracs}, but pre_split has {[len(vals) for vals in pre_split]} values."
            )

        dataset_splits_pre = [ds.sel({dim: np.asarray(vals)}) for vals in pre_split]
        ds = ds.drop_sel({dim: covered_vals})
        n_data = ds.sizes[dim]
    else:
        dataset_splits_pre = None

    if ordered:
        ds = ds.sortby(dim)
        split_idxs = [slice(offset - length, offset) for offset, length in zip(accumulate(lengths), lengths)]
        dataset_splits = [ds.isel({dim: np.asarray(idxs)}) for idxs in split_idxs]
    else:
        dataset_splits = [ds.isel({dim: np.asarray(idxs)}) for idxs in random_split(n_data, lengths, seed)]

    if dataset_splits_pre is not None:
        for i in range(len(fracs)):
            if len(pre_split[i]) > 0:
                dataset_splits[i] = xr.concat([dataset_splits[i], dataset_splits_pre[i]], dim=dim)

    return dataset_splits
