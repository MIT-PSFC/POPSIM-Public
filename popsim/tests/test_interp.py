import chex
import pytest

from popsim import interp
import numpy as np

@pytest.mark.parametrize("interp_type", [interp.InterpType.LINEAR, interp.InterpType.CUBIC, interp.InterpType.RECTILINEAR])
def test_linear_interp_time_dic(interp_type):
    tree0 = {
        "a": 1.0,
        "b": {
            "b0": 2.0,
            "b1": [3.0, -3.0],
        }
    }
    tree1 = {
        "a": 2.0,
        "b": {
            "b0": 3.0,
            "b1": [4.0, -4.0],
        }
    }
    tree2 = {
        "a": 3.0,
        "b": {
            "b0": 4.0,
            "b1": [5.0, -5.0],
        }
    }

    interpolated_tree = interp.interp_time_dic({0: tree0, 1: tree1, 2: tree2}, interp_type=interp_type)

    # Time t=0.0.
    expected0 = {
        "a": 1.0,
        "b": {
            "b0": 2.0,
            "b1": [3.0, -3.0],
        }
    }
    chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 0.0), expected0)

    # Time t=0.5.
    expected1 = {
        "a": 1.5,
        "b": {
            "b0": 2.5,
            "b1": [3.5, -3.5],
        }
    }

    # In the linear and cubic cases, the interpolated value is the average of the values at t=0 and t=1.
    # In the rectilinear case, the interpolated value is the value at t=0.
    if interp_type in [interp.InterpType.LINEAR, interp.InterpType.CUBIC]:
        chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 0.5), expected1)
    elif interp_type == interp.InterpType.RECTILINEAR:
        chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 0.5), expected0)

    # Time t=1.0
    expected2 = {
        "a": 2.0,
        "b": {
            "b0": 3.0,
            "b1": [4.0, -4.0],
        }
    }
    if interp_type in [interp.InterpType.LINEAR, interp.InterpType.CUBIC]:
        chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 1.0), expected2)
    elif interp_type == interp.InterpType.RECTILINEAR:
        chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 1.0 + np.finfo(np.float32).eps), expected2)

@pytest.mark.parametrize(
    "ts, data_input, expected_output",
    [
        # Test case 1: Simple array with NaNs in data
        (
            np.array([0, 1, 2, 3, 4, 5]),
            np.array([1.0, np.nan, 3.0, np.nan, 5.0, 6.0]),
            np.array([1.0, 1.0, 1.0, 3.0, 3.0, 5.0]),
        ),
        # Test case 2: Data is a dictionary of arrays with NaNs
        (
            np.array([0, 1, 2, 3, 4]),
            {
                'a': np.array([1.0, 1.0, np.nan, 3.0, 4.0]),
                'b': np.array([0.0, np.nan, 2.0, np.nan, 4.0]),
            },
            {
                'a': np.array([1.0, 1.0, 1.0, 1.0, 3.0]),
                'b': np.array([0.0, 0.0, 0.0, 2.0, 2.0]),
            },
        ),
        # Test case 3: Data is a list of arrays with NaNs
        (
            np.array([0, 1, 2, 3]),
            [
                np.array([1.0, 1.0, np.nan, 3.0]),
                np.array([0.0, np.nan, 2.0, np.nan]),
            ],
            [
                np.array([1.0, 1.0, 1.0, 1.0]),
                np.array([0.0, 0.0, 0.0, 2.0]),
            ],
        ),
        # Test case 4: Data with NaNs at the beginning (doesn't handle this case)
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([np.nan, np.nan, 2.0, 3.0, 3.0]),
            np.array([np.nan, np.nan, np.nan, 2.0, 3.0]),
        ),
        # Test case 5: Data with NaNs at the end
        (
            np.array([0, 1, 2, 3, 4]),
            np.array([0.0, 1.0, 2.0, np.nan, np.nan]),
            np.array([0.0, 0.0, 1.0, 2.0, 2.0]),
        ),
        # Test case 6: Data with no NaNs
        (
            np.array([0, 1, 2, 3]),
            np.array([0.0, 1.0, 2.0, 2.0]),
            np.array([0.0, 0.0, 1.0, 2.0]),
        ),
    ],
)
def test_interp_over_nans(ts, data_input, expected_output):
    result = interp.interp_over_nans(ts, data_input)
    chex.assert_trees_all_close(result, expected_output)