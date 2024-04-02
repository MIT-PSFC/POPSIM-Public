import chex
import pytest

from popsim import interp


@pytest.mark.parametrize("interp_type", ["linear", "cubic"])
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

    # Time t=1.0.
    expected1 = {
        "a": 1.5,
        "b": {
            "b0": 2.5,
            "b1": [3.5, -3.5],
        }
    }
    chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 0.5), expected1)

    # Time t=2.0.
    expected2 = {
        "a": 2.0,
        "b": {
            "b0": 3.0,
            "b1": [4.0, -4.0],
        }
    }
    chex.assert_trees_all_close(interp.resolve_paths(interpolated_tree, 1.0), expected2)
