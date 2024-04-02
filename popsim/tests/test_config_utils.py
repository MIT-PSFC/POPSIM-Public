import chex

import pytest
from jaxtyping import ArrayLike
from popsim.config_utils import CombinatorialCases, generate_combinations, build_config_paths
import popsim.enums as penums
import popsim.interp as pinterp

def test_generate_params_combinations():
    @chex.dataclass
    class Params:
        a: float
        b: float
        c: float


    # Example usage
    params = Params(a=CombinatorialCases([1.0, 2.0, 3.0]), b=2.0, c=CombinatorialCases([3.0, 4.0]))
    combinations = generate_combinations(params)

    assert len(combinations) == 6
    assert combinations[0] == Params(a=1.0, b=2.0, c=3.0)
    assert combinations[1] == Params(a=1.0, b=2.0, c=4.0)
    assert combinations[2] == Params(a=2.0, b=2.0, c=3.0)
    assert combinations[3] == Params(a=2.0, b=2.0, c=4.0)
    assert combinations[4] == Params(a=3.0, b=2.0, c=3.0)
    assert combinations[5] == Params(a=3.0, b=2.0, c=4.0)

@pytest.mark.parametrize("interp_type", ["linear", "cubic"])
def test_build_config_paths(interp_type):
    @chex.dataclass
    class Params:
        a: float
        nested_b: dict[str, ArrayLike]
        imps: dict[penums.Impurity, float]

    config = Params(
        a=1.0,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )

    #
    # Check that running "build_config_paths" on this config does not do anything.
    #
    new_config = build_config_paths(config, interp_type)
    chex.assert_trees_all_equal(new_config, config)

    #
    # Now have the user specify a time-dependent impurity sequence.
    #
    time_dep_imps = {
        0.0: {
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
        1.0: {
            penums.Impurity.Tungsten: 5.0,
            penums.Impurity.Neon: 6.0,
        },
        2.5: {
            penums.Impurity.Tungsten: 6.0,
            penums.Impurity.Neon: 7.0,
        },
    }
    config.imps = time_dep_imps
    new_config2 = build_config_paths(config, interp_type)
    interped_imps = pinterp.interp_time_dic(time_dep_imps, interp_type)
    expected_config = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=interped_imps)
    chex.assert_trees_all_equal(new_config2, expected_config)

    #
    # Now have the user specify a time-dependent impurity sequence but only for Tungsten.
    #
    config.imps = {
        penums.Impurity.Tungsten: {0.0: 4.0, 1.0: 5.0, 2.5: 6.0},
        penums.Impurity.Neon: 5.0,
    }
    new_config3 = build_config_paths(config, interp_type)
    expected_config3 = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: interped_imps[penums.Impurity.Tungsten], penums.Impurity.Neon: 5.0})
    chex.assert_trees_all_equal(new_config3, expected_config3)