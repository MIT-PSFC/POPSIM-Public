import math
import random

import jax
import pytest
import jax.numpy as jnp
import xarray as xr

from popsim.ml.split_utils import fracs_to_lengths, random_split, split_dataset_by_fracs, split_dataset_by_vals
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

def test_split_dataset_by_fracs(cmod_test_dataset):
    ds = cmod_test_dataset
    split_fracs = [0.68, 0.21, 0.11]
    split_datasets = split_dataset_by_fracs(ds, split_fracs, "shot", 42)
    
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
        split_dataset_by_fracs(ds, [0.5, 0.5, 0.01], "shot", 42)


def test_split_dataset_by_vals(cmod_test_dataset):
    ds = cmod_test_dataset

    # Check that the lengths of the split dataset list is correct and the reduced dataset does not contain the values specified in the split_vals
    split_vals_idxs = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]]
    split_vals = [ds["shot"].values[idxs] for idxs in split_vals_idxs]
    dataset_splits, reduced_dataset = split_dataset_by_vals(ds, split_vals, "shot")
    assert len(dataset_splits) == len(split_vals)
    for shot in jnp.concatenate(split_vals):
        assert shot not in reduced_dataset["shot"].values

    # Check than an error is thrown if the pre split doesn't contain values that are in the dataset
    split_vals = [[-1], [], []]
    with pytest.raises(ValueError):
        split_dataset_by_vals(ds, split_vals, "shot")

@pytest.mark.parametrize("pre_split_idxs", [None, [[], [], []], [[0, 10, 20], [], []], [[1], [3], [2]]])
@pytest.mark.parametrize("ordered", [True, False])
def test_split_dataset_pre_split_and_ordered(cmod_test_dataset: xr.Dataset, pre_split_idxs, ordered):
    ds = cmod_test_dataset

    if pre_split_idxs:
        pre_split_vals = [ds["shot"].values[idxs] for idxs in pre_split_idxs]
    else:
        pre_split_vals = None

    split_fracs = [0.68, 0.21, 0.11]

    def check_splits_ordered(split_datasets, dim, pre_split_vals):
        # Ordered except for the pre_split values
        if pre_split_vals:
            for i in range(len(pre_split_vals)):
                if len(pre_split_vals[i]) > 0:
                    split_datasets[i] = split_datasets[i].drop_sel({dim: jnp.asarray(pre_split_vals[i])})

        for i in range(1, len(split_datasets)):
            max_prev = split_datasets[i - 1][dim].values.max()
            min_curr = split_datasets[i][dim].values.min()
            assert max_prev < min_curr

    def check_splits_contain_pre_split_vals(split_datasets, pre_split_vals):
        for split, split_vals in zip(split_datasets, pre_split_vals):
            assert jnp.all(jnp.isin(split_vals, split["shot"].values))

    if pre_split_vals:
        pre_split_datasets, reduced_dataset = split_dataset_by_vals(ds, pre_split_vals, "shot")
        random_split_datasets = split_dataset_by_fracs(reduced_dataset, split_fracs, "shot", 42, ordered=ordered)
        split_datasets = [
            xr.concat([pre_split_dataset, random_split_dataset], dim="shot") if len(pre_split_dataset["shot"]) > 0 else random_split_dataset
            for pre_split_dataset, random_split_dataset in zip(pre_split_datasets, random_split_datasets)
        ]
    else:
        split_datasets = split_dataset_by_fracs(ds, split_fracs, "shot", 42, ordered=ordered)

    num_shots_original = len(ds["shot"])
    num_shots_split = sum(len(split["shot"]) for split in split_datasets)
    assert num_shots_original == num_shots_split

    if pre_split_vals:
        check_splits_contain_pre_split_vals(split_datasets, pre_split_vals)
    if ordered:
        check_splits_ordered(split_datasets, "shot", pre_split_vals)