import math
import random

import jax
import pytest
import jax.numpy as jnp
import xarray as xr

from popsim.ml.split_utils import fracs_to_lengths, random_split, split_dataset_along_dim
from popsim.tests.fixtures import cmod_test_dataset

def test_fracs_to_lengths():
    def generate_three_numbers_sum_to_one():
        # Generate two random numbers between 0 and 1
        first, second = random.random(), random.random()

        # Ensure they are in ascending order
        first, second = sorted([first, second])

        # Calculate the three segments
        a = first
        b = second - first
        c = 1 - second

        return a, b, c

    n_fracs_try = 100
    lengths_to_try = [2**i for i in range(32)]
    for _ in range(n_fracs_try):
        fracs = generate_three_numbers_sum_to_one()
        assert math.isclose(sum(fracs), 1.0)

        for total_length in lengths_to_try:
            lengths = fracs_to_lengths(total_length, fracs)
            assert sum(lengths) == total_length
            assert len(lengths) == 3
            assert all(isinstance(le, int) for le in lengths)
            assert all(
                abs(length - round(total_length * frac)) <= 1
                for length, frac in zip(lengths, fracs)
            )


def test_random_split():
    n_data = 10
    fracs = [0.7, 0.2, 0.1]
    key = 42
    idx_sets = random_split(n_data, fracs, key)
    for frac, idx_set in zip(fracs, idx_sets):
        assert round(frac * n_data) - len(idx_set) <= 1

    concatenated_idxs = jnp.concatenate(idx_sets)
    assert len(concatenated_idxs) == n_data
    assert jnp.all(jnp.sort(concatenated_idxs) == jnp.arange(n_data))
    assert len(idx_sets[0]) == 7
    assert len(idx_sets[1]) == 2
    assert len(idx_sets[2]) == 1

    lengths = [7, 2, 1]
    idx_sets = random_split(n_data, lengths, key)
    assert len(idx_sets[0]) == lengths[0]
    assert len(idx_sets[1]) == lengths[1]
    assert len(idx_sets[2]) == lengths[2]

def test_split_dataset_along_dim(cmod_test_dataset):
    ds = cmod_test_dataset
    split_fracs = [0.68, 0.21, 0.11]
    split_datasets = split_dataset_along_dim(ds, split_fracs, "shot", 42)
    
    assert len(split_datasets) == 3
    # Check that the lengths of the splits are correct
    n_shots = len(ds["shot"])
    expected_lengths = [int(n_shots * frac) for frac in split_fracs]
    for split, expected_length in zip(split_datasets, expected_lengths):
        # Allow for differences of 1 due to rounding.
        assert abs(len(split["shot"]) - expected_length) <= 1

    # Check that the splits are disjoint
    all_shots = jnp.unique(jnp.concatenate([split["shot"].values for split in split_datasets]))
    assert len(all_shots) == n_shots

    # Check that the function errors if the split fractions do not sum to 1
    with pytest.raises(ValueError):
        split_dataset_along_dim(ds, [0.5, 0.5, 0.01], "shot", 42)

    # Check that an error is thrown if the pre split length doesn't match the split fracs length
    pre_split = [[0, 10, 20], []]
    with pytest.raises(ValueError):
        split_dataset_along_dim(ds, split_fracs, "shot", 42, pre_split=pre_split)

    # Check than an error is thrown if the pre split doesn't contain dims that are in the dataset
    pre_split = [[-1], [], []]
    with pytest.raises(ValueError):
        split_dataset_along_dim(ds, split_fracs, "shot", 42, pre_split=pre_split)


@pytest.mark.parametrize("pre_split_idxs", [None, [[], [], []], [[0, 10, 20], [], []], [[1], [3], [2]]])
@pytest.mark.parametrize("ordered", [True, False])
def test_split_dataset_along_dim_ordered_and_pre_split(cmod_test_dataset: xr.Dataset, pre_split_idxs, ordered):
    ds = cmod_test_dataset

    if pre_split_idxs is None:
        pre_split = None
    else:
        pre_split = [ds["shot"].values[idxs] for idxs in pre_split_idxs]

    split_fracs = [0.68, 0.21, 0.11]

    def check_splits_ordered(split_datasets, dim, pre_split):
        if pre_split:
            for split_dataset, pre_split_vals in zip(split_datasets, pre_split):
                if len(pre_split_vals) > 0:
                    split_dataset = split_dataset.drop_sel({dim: jnp.asarray(pre_split_vals)})

        for i in range(1, len(split_datasets)):
            max_prev = split_datasets[i - 1][dim].values.max()
            min_curr = split_datasets[i][dim].values.min()
            assert max_prev < min_curr

    def check_splits_contain_pre_split(split_datasets, pre_split):
        for split, pre_split in zip(split_datasets, pre_split):
            assert jnp.all(jnp.isin(pre_split, split["shot"].values))

    split_datasets = split_dataset_along_dim(ds, split_fracs, "shot", 42, ordered=ordered, pre_split=pre_split)
    if pre_split:
        check_splits_contain_pre_split(split_datasets, pre_split)
    if ordered:
        check_splits_ordered(split_datasets, "shot", pre_split)