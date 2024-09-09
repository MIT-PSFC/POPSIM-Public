import math
from collections.abc import Sequence
from itertools import accumulate
from typing import Union

import jax
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


def random_split(
    n_data: int,
    lengths_or_fracs: Sequence[Union[int, float]],
    key: jax.random.PRNGKey,
) -> list[jax.Array]:
    """Generate indicies to split a dataset into non-overlapping new datasets.

    Args:
        n_data (int): size of the dataset to be split.
        lengths_or_fracs (Sequence[Union[int, float]]): lengths or fractions of splits to be produced.
        key (jax.random.PRNGKey): key to use for random number generation.

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
        key,
        n_data,
    )
    return [indices[offset - length : offset] for offset, length in zip(accumulate(lengths), lengths)]


def split_dataset_by_coords(ds: xr.Dataset, fracs: Sequence[float], coord: str, key: jax.random.PRNGKey) -> Sequence[xr.Dataset]:
    """Split a dataset into non-overlapping new datasets based on a coordinate.

    Args:
        ds (xr.Dataset): dataset to be split.
        fracs (Sequence[float]): fractions of splits to be produced.
        coord (str): coordinate to split the dataset by.
        key (jax.random.PRNGKey): key to use for random number generation.

    Returns:
        Sequence[xr.Dataset]: sequence of datasets.
    """
    n_data = len(ds[coord])
    lengths = fracs_to_lengths(n_data, fracs)
    return [ds.isel({coord: idxs}) for idxs in random_split(n_data, lengths, key)]
