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
@pytest.mark.parametrize("sortby", [None, 'shot', 'beta_p'])
def test_split_dataset_pre_split_and_sorted(cmod_test_dataset: xr.Dataset, pre_split_idxs, sortby):
    ds = cmod_test_dataset
    episode_coord = "shot"

    if pre_split_idxs:
        pre_split_vals = [ds[episode_coord].values[idxs] for idxs in pre_split_idxs]
    else:
        pre_split_vals = None

    split_fracs = [0.68, 0.21, 0.11]

    def check_splits_sorted(split_datasets, sortby, pre_split_vals):
        # Sorted except for the pre_split values
        if pre_split_vals:
            for i in range(len(pre_split_vals)):
                if len(pre_split_vals[i]) > 0:
                    split_datasets[i] = split_datasets[i].drop_sel({episode_coord: jnp.asarray(pre_split_vals[i])})

        for i in range(1, len(split_datasets)):
            # To ensure the splits are sorted correctly, compare the following two values:
            # The highest maximum value of the previous split (just the max)
            # The lowest maximum value of the current split
            max_prev = jnp.nanmax(split_datasets[i - 1][sortby].values)
            curr_maxes = split_datasets[i][sortby].max(dim=[d for d in split_datasets[i][sortby].dims if d != episode_coord], skipna=True)
            min_of_curr_maxes = jnp.nanmin(curr_maxes.values)
            
            assert max_prev <= min_of_curr_maxes

    def check_splits_contain_pre_split_vals(split_datasets, pre_split_vals):
        for split, split_vals in zip(split_datasets, pre_split_vals):
            assert jnp.all(jnp.isin(split_vals, split[episode_coord].values))

    if pre_split_vals:
        pre_split_datasets, reduced_dataset = split_dataset_by_vals(ds, pre_split_vals, episode_coord)
        random_split_datasets = split_dataset_by_fracs(reduced_dataset, split_fracs, episode_coord, 42, sortby=sortby)
        split_datasets = [
            xr.concat([pre_split_dataset, random_split_dataset], dim=episode_coord) if len(pre_split_dataset[episode_coord]) > 0 else random_split_dataset
            for pre_split_dataset, random_split_dataset in zip(pre_split_datasets, random_split_datasets)
        ]
    else:
        split_datasets = split_dataset_by_fracs(ds, split_fracs, episode_coord, 42, sortby=sortby)

    num_episodes_original = len(ds[episode_coord])
    num_episodes_split = sum(len(split[episode_coord]) for split in split_datasets)
    assert num_episodes_original == num_episodes_split

    if pre_split_vals:
        check_splits_contain_pre_split_vals(split_datasets, pre_split_vals)
    if sortby:
        check_splits_sorted(split_datasets, sortby, pre_split_vals)