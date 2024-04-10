import dataclasses

import chex
import pytest
import jax
from jaxtyping import ArrayLike

import popsim.enums as penums
import popsim.interp as pinterp
from popsim.param_utils import (
    CombinatorialCases,
    MultiCases,
    build_config_paths,
    check_config,
    generate_combinatorial_cases,
    generate_multi_cases,
    build_configs,
    make_time_base
)


@chex.dataclass
class SimpleParams:
    a: float
    b: float
    c: float

def test_generate_params_combinations():
    #
    # Generate combinations of simple objects (floats) while keeping a dictionary constant.
    #
    static_b = SimpleParams(a=1.0, b={"b0": 2.0, "b1": [3.0, -3.0]}, c=3.0)
    params = SimpleParams(a=CombinatorialCases(cases=[1.0, 2.0, 3.0]), b=static_b, c=CombinatorialCases(cases=[{'a': 3.0}, {'a': 4.0}]))
    combinations = generate_combinatorial_cases(params)

    assert len(combinations) == 6
    assert combinations[0] == SimpleParams(a=1.0, b=params.b, c={'a': 3.0})
    assert combinations[1] == SimpleParams(a=1.0, b=params.b, c={'a': 4.0})
    assert combinations[2] == SimpleParams(a=2.0, b=params.b, c={'a': 3.0})
    assert combinations[3] == SimpleParams(a=2.0, b=params.b, c={'a': 4.0})
    assert combinations[4] == SimpleParams(a=3.0, b=params.b, c={'a': 3.0})
    assert combinations[5] == SimpleParams(a=3.0, b=params.b, c={'a': 4.0})

    #
    # Generate combinations of more complex objects (nested dicts, lists).
    #
    cases_a = CombinatorialCases(cases=[{"a": {"a0": 1.0, "a1": 2.0}}, {"a": {"a0": 3.0, "a1": 4.0}}])
    cases_b = CombinatorialCases(cases=[[1.0, 2.0], [3.0, 4.0]])
    cases_c = CombinatorialCases(cases=[{0.0: 2.0, 1.0: 3.0}, {0.0: 4.0, 1.0: 5.0}])
    params2 = SimpleParams(a=cases_a, b=cases_b, c=cases_c)
    combinations2 = generate_combinatorial_cases(params2)
    assert len(combinations2) == len(cases_a.cases) * len(cases_b.cases) * len(cases_c.cases)
    assert combinations2[0] == SimpleParams(a=cases_a.cases[0], b=cases_b.cases[0], c=cases_c.cases[0])
    assert combinations2[1] == SimpleParams(a=cases_a.cases[0], b=cases_b.cases[0], c=cases_c.cases[1])
    assert combinations2[2] == SimpleParams(a=cases_a.cases[0], b=cases_b.cases[1], c=cases_c.cases[0])
    assert combinations2[3] == SimpleParams(a=cases_a.cases[0], b=cases_b.cases[1], c=cases_c.cases[1])
    assert combinations2[4] == SimpleParams(a=cases_a.cases[1], b=cases_b.cases[0], c=cases_c.cases[0])
    assert combinations2[5] == SimpleParams(a=cases_a.cases[1], b=cases_b.cases[0], c=cases_c.cases[1])
    assert combinations2[6] == SimpleParams(a=cases_a.cases[1], b=cases_b.cases[1], c=cases_c.cases[0])
    assert combinations2[7] == SimpleParams(a=cases_a.cases[1], b=cases_b.cases[1], c=cases_c.cases[1])

def test_generate_multi_cases():
    params = SimpleParams(a=MultiCases(cases=[1.0, 2.0]), b=[2.0, 3.0], c=MultiCases(cases=[{'a': 1.0}, {'a': 2.0}]))
    cases = generate_multi_cases(params)
    assert len(cases) == 2
    chex.assert_trees_all_equal(cases[0], SimpleParams(a=1.0, b=[2.0, 3.0], c={'a': 1.0}))
    chex.assert_trees_all_equal(cases[1], SimpleParams(a=2.0, b=[2.0, 3.0], c={'a': 2.0}))

    params2 = SimpleParams(a={"a": -1.0}, b=[2.0, 3.0], c=3.0)
    cases2 = generate_multi_cases(params2)
    chex.assert_trees_all_equal(params2, cases2)

# Note: cubic case currently breaks as extrapolation of the time dictionary is not handeled as expected.
@pytest.mark.parametrize("interp_type", ["linear"])
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
    time_base = make_time_base(0.0, 10.0, 0.1)

    #
    # Should return interps that are constant. Spot check at start and end.
    #
    new_config = build_config_paths(config, time_base, interp_type)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config, time_base[0]), config)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config, time_base[-1]), config)

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
    new_config2 = build_config_paths(config, time_base, interp_type)
    interped_imps = pinterp.interp_time_dic(time_dep_imps, interp_type)
    expected_config_begin = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=time_dep_imps[0.0])
    expected_config_t25 = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=time_dep_imps[2.5])

    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config2, 0.0), expected_config_begin)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config2, 2.5), expected_config_t25)


    #
    # Now have the user specify a time-dependent impurity sequence but only for Tungsten.
    #
    config.imps = {
        penums.Impurity.Tungsten: {0.0: 4.0, 1.0: 5.0, 2.5: 6.0},
        penums.Impurity.Neon: 5.0,
    }
    new_config3 = build_config_paths(config, time_base, interp_type)
    expected_config3_begin = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 4.0, penums.Impurity.Neon: 5.0})
    expected_config_3_t25 = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 6.0, penums.Impurity.Neon: 5.0})
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config3, 0.0), expected_config3_begin)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config3, 2.5), expected_config_3_t25)

    #
    # Test that this works with MultiCases.
    #
    time_dep_tungsten0, time_dep_tungsten1 = {0.0: 4.0, 1.0: 5.0}, {0.0: 4.0, 1.0: 6.0, 2.5: 7.0}
    interp_tungsten0, interp_tungsten1 = pinterp.interp_time_dic(time_dep_tungsten0, interp_type), pinterp.interp_time_dic(time_dep_tungsten1, interp_type)
    config.imps = {
        penums.Impurity.Tungsten: MultiCases(cases=[time_dep_tungsten0, time_dep_tungsten1]),
        penums.Impurity.Neon: MultiCases(cases=[5.0, 6.0]),
    }
    new_config4 = build_config_paths(config, time_base, interp_type)
    expected_config_4_t1 = Params(
        a=1.0,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={penums.Impurity.Tungsten: MultiCases(cases=[5.0, 6.0]), penums.Impurity.Neon: MultiCases(cases=[5.0, 6.0])}
    )
    expected_config4_t25 = Params(
        a=1.0,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={penums.Impurity.Tungsten: MultiCases(cases=[5.0, 7.0]), penums.Impurity.Neon: MultiCases(cases=[5.0, 6.0])}
    )
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config4, 1.0), expected_config_4_t1)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config4, 2.5), expected_config4_t25)

    #
    # Test that this works with CombinatorialCases.
    #
    config.imps = {
        penums.Impurity.Tungsten: CombinatorialCases(cases=[time_dep_tungsten0, time_dep_tungsten1]),
        penums.Impurity.Neon: CombinatorialCases(cases=[5.0, 6.0]),
    }
    new_config5 = build_config_paths(config, time_base, interp_type)

    assert len(new_config5.imps[penums.Impurity.Tungsten].cases) == 2
    assert len(new_config5.imps[penums.Impurity.Neon].cases) == 2

    expected_config5_t25 = Params(
        a=1.0,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={penums.Impurity.Tungsten: CombinatorialCases(cases=[5.0, 7.0]), penums.Impurity.Neon: CombinatorialCases(cases=[5.0, 6.0])}
    )
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_config5, 2.5), expected_config5_t25)

@pytest.mark.parametrize("config, should_raise", [
    # Neither CombinatorialCases nor MultiCases, should not raise an exception
    ({"example": SimpleParams(a=1.0, b=2.0, c=3.0)}, False),

    # Only MultiCases, should not raise an exception
    ({"example": SimpleParams(a=MultiCases(cases=[1.0, 2.0]), b=2.0, c=3.0)}, False),

    # Only CombinatorialCases, should not raise an exception
    ({"example": SimpleParams(a=CombinatorialCases(cases=[1.0, 2.0]), b=2.0, c=CombinatorialCases(cases=[3.0, 4.0]))}, False),

    # Both CombinatorialCases and MultiCases, should raise an exception
    ({"example": SimpleParams(a=CombinatorialCases(cases=[1.0, 2.0]), b=2.0, c=MultiCases(cases=[3.0, 4.0]))}, True),

    # Only MultiCases, but lengths are not the same, should raise an exception.
    ({"example": SimpleParams(a=MultiCases(cases=[1.0, 2.0]), b=2.0, c=MultiCases(cases=[3.0, 4.0, 5.0]))}, True),
])
def test_check_config(config, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            check_config(config)
    else:
        # This block attempts to run check_config and will fail the test if an exception is raised
        check_config(config)

@pytest.mark.parametrize("interp_type", ["linear", "cubic"])
def test_build_config(interp_type):
    """
    Test the user-facing API for building configurations.
    We want to cover:
        1) The identity case (output config should be exact same).
        2) Interpolation in the linear and cubic cases and with both MultiCases and CombinatorialCases.
        3) Complex data structures
    """
    @chex.dataclass
    class Params:
        a: float
        nested_b: dict[str, ArrayLike]
        imps: dict[penums.Impurity, float]
    time_base = make_time_base(0.0, 10.0, 0.1)

    #
    # Check the identity case.
    #
    config = Params(
        a=1.0,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )
    new = build_configs(config, time_base, interp_type)
    def check_for_time(t):
        chex.assert_trees_all_equal(pinterp.resolve_paths(new, t), config)
    check_for_time(0.0)
    check_for_time(5.0)
    check_for_time(10.0)

    #
    # Now specify a trajectory for Tungsten.
    #
    time_dep_tungsten = {
        0.0: 4.0,
        1.0: 5.0,
        2.5: 6.0,
        10.0: 6.0
    }
    config.imps[penums.Impurity.Tungsten] = time_dep_tungsten
    interped_tungsten = pinterp.interp_time_dic(time_dep_tungsten, interp_type)
    def expected_config_for_time(t):
        expected_config = dataclasses.replace(config, imps={penums.Impurity.Tungsten: time_dep_tungsten[t], penums.Impurity.Neon: 5.0})
        return expected_config
    built_config = build_configs(config, time_base, interp_type)

    chex.assert_trees_all_equal(pinterp.resolve_paths(built_config, 0.0), expected_config_for_time(0.0))
    chex.assert_trees_all_equal(pinterp.resolve_paths(built_config, 1.0), expected_config_for_time(1.0))
    chex.assert_trees_all_equal(pinterp.resolve_paths(built_config, 2.5), expected_config_for_time(2.5))

    #
    # Now use MultiCases to generate the following cases:
    #   (interped_tungsten, constant_neon)
    #   (constant_tungsten, interped_neon)
    #   (interped_tungsten, interped_neon)
    #
    time_dep_neon = {
        0.0: 5.0,
        1.0: 6.0,
        2.5: 7.0,
    }
    interped_neon = pinterp.interp_time_dic(time_dep_neon, interp_type)
    config.imps = {
        penums.Impurity.Tungsten: MultiCases(cases=[time_dep_tungsten, 4.0, time_dep_tungsten]),
        penums.Impurity.Neon: MultiCases(cases=[5.0, time_dep_neon, time_dep_neon]),
    }
    cases = build_configs(config, time_base, interp_type)

    assert len(cases) == 3

    expected_imps_at_t25_for_cases = [
        {penums.Impurity.Tungsten: time_dep_tungsten[2.5], penums.Impurity.Neon: 5.0},
        {penums.Impurity.Tungsten: 4.0, penums.Impurity.Neon: time_dep_neon[2.5]},
        {penums.Impurity.Tungsten: time_dep_tungsten[2.5], penums.Impurity.Neon: time_dep_neon[2.5]},
    ]
    for case, expected_imps in zip(cases, expected_imps_at_t25_for_cases):
        case_resolved = pinterp.resolve_paths(case, 2.5)
        chex.assert_trees_all_equal(case_resolved.imps, expected_imps)
    

    #
    # Now use combinatorial cases to test multiple impurity cases simultaneously with different values of "b0".
    #
    b0_cases = CombinatorialCases(cases=[1.0, (2.0, 3.0)])
    tungsten_cases = CombinatorialCases(cases=[time_dep_tungsten, 4.0])
    config = Params(
        a=1.0,
        nested_b={"b0": b0_cases, "b1": [3.0, -3.0]},
        imps={
            penums.Impurity.Tungsten: tungsten_cases,
            penums.Impurity.Neon: 5.0,
        },
    )
    cases = build_configs(config, time_base, interp_type)
    assert len(cases) == len(b0_cases.cases) * len(tungsten_cases.cases)

    ref_case = cases[0] # All cases should look like case[0] expect where we will replace things.


    expected_cases = [
        dataclasses.replace(ref_case, nested_b={"b0": 1.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: interped_tungsten, penums.Impurity.Neon: 5.0}),
        dataclasses.replace(ref_case, nested_b={"b0": (2.0, 3.0), "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: interped_tungsten, penums.Impurity.Neon: 5.0}),
        dataclasses.replace(ref_case, nested_b={"b0": 1.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 4.0, penums.Impurity.Neon: 5.0}),
        dataclasses.replace(ref_case, nested_b={"b0": (2.0, 3.0), "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 4.0, penums.Impurity.Neon: 5.0}),
    ]

    # Why did I copy-paste this? Well chex.chexify (which is needed to put a chex assertion into a function)
    # was being exceptionally annoying in not letting me put this stuff into a function.
    for case, expected_case in zip(cases, expected_cases):
        resolved_case = pinterp.resolve_paths(case, 0.0)
        expected_case_resolved = pinterp.resolve_paths(expected_case, 0.0)
        chex.assert_trees_all_equal(resolved_case, expected_case_resolved)
    
    for case, expected_case in zip(cases, expected_cases):
        resolved_case = pinterp.resolve_paths(case, 2.5)
        expected_case_resolved = pinterp.resolve_paths(expected_case, 2.5)
        chex.assert_trees_all_equal(resolved_case, expected_case_resolved)

    for case, expected_case in zip(cases, expected_cases):
        resolved_case = pinterp.resolve_paths(case, 9.0)
        expected_case_resolved = pinterp.resolve_paths(expected_case, 9.0)
        chex.assert_trees_all_equal(resolved_case, expected_case_resolved)
    
    #
    # Check the case where a config is an interp.
    #
    interped_a = pinterp.interp(time_base, time_base, interp_type=interp_type)
    config = Params(
        a=interped_a,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )
    new = build_configs(config, time_base, interp_type)

    expected_t0 = dataclasses.replace(config, a=0.0)
    expected_t15 = dataclasses.replace(config, a=1.5)
    expected_t812 = dataclasses.replace(config, a=8.12)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new, 0.0), expected_t0)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new, 1.5), expected_t15)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new, 8.12), expected_t812)