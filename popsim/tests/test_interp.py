import chex
import popsim.interp as interp

def test_linear_interp_time_dic():
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

    interpolated_tree = interp.interp_time_dic({0: tree0, 1: tree1, 2: tree2})
    expected0 = {
        "a": 1.5,
        "b": {
            "b0": 2.5,
            "b1": [3.5, -3.5],
        }
    }
    chex.assert_trees_all_close(interpolated_tree.evaluate(0.5), expected0)