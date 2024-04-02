import dataclasses

import chex
import pytest
from jaxtyping import ArrayLike

import popsim.enums as penums
import popsim.interp as pinterp
from popsim.config_utils import (
    CombinatorialCases,
    MultiCases,
    build_config_paths,
    check_config,
    generate_combinatorial_cases,
    generate_multi_cases
)


@chex.dataclass
class SimpleParams:
    a: float
    b: float
    c: float

def test_generate_params_combinations():
    # Example usage
    complex_b = SimpleParams(a=1.0, b={"b0": 2.0, "b1": [3.0, -3.0]}, c=3.0)
    params = SimpleParams(a=CombinatorialCases(config=[1.0, 2.0, 3.0]), b=complex_b, c=CombinatorialCases(config=[{'a': 3.0}, {'a': 4.0}]))
    combinations = generate_combinatorial_cases(params)

    assert len(combinations) == 6
    assert combinations[0] == SimpleParams(a=1.0, b=params.b, c={'a': 3.0})
    assert combinations[1] == SimpleParams(a=1.0, b=params.b, c={'a': 4.0})
    assert combinations[2] == SimpleParams(a=2.0, b=params.b, c={'a': 3.0})
    assert combinations[3] == SimpleParams(a=2.0, b=params.b, c={'a': 4.0})
    assert combinations[4] == SimpleParams(a=3.0, b=params.b, c={'a': 3.0})
    assert combinations[5] == SimpleParams(a=3.0, b=params.b, c={'a': 4.0})

def test_generate_multi_cases():
    params = SimpleParams(a=MultiCases(config=[1.0, 2.0]), b=[2.0, 3.0], c=MultiCases(config=[{'a': 1.0}, {'a': 2.0}]))
    cases = generate_multi_cases(params)
    assert len(cases) == 2
    chex.assert_trees_all_equal(cases[0], SimpleParams(a=1.0, b=[2.0, 3.0], c={'a': 1.0}))
    chex.assert_trees_all_equal(cases[1], SimpleParams(a=2.0, b=[2.0, 3.0], c={'a': 2.0}))

    params2 = SimpleParams(a={"a": -1.0}, b=[2.0, 3.0], c=3.0)
    cases2 = generate_multi_cases(params2)
    chex.assert_trees_all_equal(params2, cases2)

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

    #
    # Test that this works with MultiCases.
    #
    time_dep_tungsten0, time_dep_tungsten1 = {0.0: 4.0, 1.0: 5.0}, {0.0: 4.0, 1.0: 6.0}
    interp_tungsten0, interp_tungsten1 = pinterp.interp_time_dic(time_dep_tungsten0, interp_type), pinterp.interp_time_dic(time_dep_tungsten1, interp_type)
    config.imps = {
        penums.Impurity.Tungsten: MultiCases(config=[time_dep_tungsten0, time_dep_tungsten1]),
        penums.Impurity.Neon: MultiCases(config=[5.0, 6.0]),
    }
    new_config4 = build_config_paths(config, interp_type)
    expected_config4 = dataclasses.replace(
        config,
        imps={penums.Impurity.Tungsten: MultiCases(config=[interp_tungsten0, interp_tungsten1]), penums.Impurity.Neon: MultiCases(config=[5.0, 6.0])},
    )
    chex.assert_trees_all_equal(new_config4, expected_config4)

    #
    # Test that this works with CombinatorialCases.
    #
    config.imps = {
        penums.Impurity.Tungsten: CombinatorialCases(config=[time_dep_tungsten0, time_dep_tungsten1]),
        penums.Impurity.Neon: CombinatorialCases(config=[5.0, 6.0]),
    }
    new_config5 = build_config_paths(config, interp_type)
    expected_config5 = dataclasses.replace(
        config,
        imps={penums.Impurity.Tungsten: CombinatorialCases(config=[interp_tungsten0, interp_tungsten1]), penums.Impurity.Neon: CombinatorialCases(config=[5.0, 6.0])},
    )
    chex.assert_trees_all_equal(new_config5, expected_config5)

@pytest.mark.parametrize("config, should_raise", [
    # Neither CombinatorialCases nor MultiCases, should not raise an exception
    ({"example": SimpleParams(a=1.0, b=2.0, c=3.0)}, False),

    # Only MultiCases, should not raise an exception
    ({"example": SimpleParams(a=MultiCases(config=[1.0, 2.0]), b=2.0, c=3.0)}, False),

    # Only CombinatorialCases, should not raise an exception
    ({"example": SimpleParams(a=CombinatorialCases(config=[1.0, 2.0]), b=2.0, c=CombinatorialCases(config=[3.0, 4.0]))}, False),

    # Both CombinatorialCases and MultiCases, should raise an exception
    ({"example": SimpleParams(a=CombinatorialCases(config=[1.0, 2.0]), b=2.0, c=MultiCases(config=[3.0, 4.0]))}, True),

    # Only MultiCases, but lengths are not the same, should raise an exception.
    ({"example": SimpleParams(a=MultiCases(config=[1.0, 2.0]), b=2.0, c=MultiCases(config=[3.0, 4.0, 5.0]))}, True),
])
def test_check_config(config, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            check_config(config)
    else:
        # This block attempts to run check_config and will fail the test if an exception is raised
        check_config(config)
