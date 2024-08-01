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