import dataclasses

import chex
import pytest
from jaxtyping import ArrayLike
import numpy as np

import popsim.enums as penums
import popsim.interp as pinterp
from popsim.simulate import make_time_base, SimInput
from popsim.param_utils import (
    build_param_paths,
    param_specs_to_paths
)


@chex.dataclass
class SimpleParams:
    a: float
    b: float
    c: float


# Note: cubic case currently breaks as extrapolation of the time dictionary is not handeled as expected.
@pytest.mark.parametrize("interp_type", [pinterp.InterpType.LINEAR])
def test_build_param_paths(interp_type):
    @chex.dataclass
    class Params:
        a: float
        nested_b: dict[str, ArrayLike]
        imps: dict[penums.Impurity, float]

    params = Params(
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
    new_params = build_param_paths(params, time_base, interp_type)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_params, time_base[0]), params)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_params, time_base[-1]), params)

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
    params.imps = time_dep_imps
    new_params2 = build_param_paths(params, time_base, interp_type)
    interped_imps = pinterp.interp_time_dic(time_dep_imps, interp_type)
    expected_params_begin = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=time_dep_imps[0.0])
    expected_params_t25 = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=time_dep_imps[2.5])

    chex.assert_trees_all_equal(pinterp.resolve_paths(new_params2, 0.0), expected_params_begin)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_params2, 2.5), expected_params_t25)


    #
    # Now have the user specify a time-dependent impurity sequence but only for Tungsten.
    #
    params.imps = {
        penums.Impurity.Tungsten: {0.0: 4.0, 1.0: 5.0, 2.5: 6.0},
        penums.Impurity.Neon: 5.0,
    }
    new_params3 = build_param_paths(params, time_base, interp_type)
    expected_params3_begin = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 4.0, penums.Impurity.Neon: 5.0})
    expected_params_3_t25 = Params(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 6.0, penums.Impurity.Neon: 5.0})
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_params3, 0.0), expected_params3_begin)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new_params3, 2.5), expected_params_3_t25)


    #
    # Check that we can specify a time-dependent subtree.
    #
    time_dep_b = {
        0.0: {"b0": 2.0, "b1": [3.0, -3.0]},
        1.0: {"b0": 3.0, "b1": [4.0, -4.0]},
        2.5: {"b0": 4.0, "b1": [5.0, -5.0]},
    }
    time_dep_params = Params(
        a=1.0,
        nested_b=time_dep_b,
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )
    built_params = build_param_paths(time_dep_params, time_base, interp_type)

    b_at_zero = pinterp.resolve_paths(built_params.nested_b, 0.0)
    b_at_point_five = pinterp.resolve_paths(built_params.nested_b, 0.5)
    b_at_one = pinterp.resolve_paths(built_params.nested_b, 1.0)
    b_at_25 = pinterp.resolve_paths(built_params.nested_b, 2.5)

    if interp_type == "linear":
        expected_at_point_five = {"b0": 2.5, "b1": [3.5, -3.5]}
        chex.assert_trees_all_equal(b_at_point_five, expected_at_point_five)

    chex.assert_trees_all_equal(b_at_zero, time_dep_b[0.0])
    chex.assert_trees_all_equal(b_at_one, time_dep_b[1.0])
    chex.assert_trees_all_equal(b_at_25, time_dep_b[2.5])


    #
    # Check the case where a params is an interp.
    #
    interped_a = pinterp.interp(time_base, time_base, interp_type=interp_type)
    params = Params(
        a=interped_a,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )
    new = build_param_paths(params, time_base, interp_type)

    expected_t0 = dataclasses.replace(params, a=0.0)
    expected_t15 = dataclasses.replace(params, a=1.5)
    expected_t812 = dataclasses.replace(params, a=8.12)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new, 0.0), expected_t0)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new, 1.5), expected_t15)
    chex.assert_trees_all_equal(pinterp.resolve_paths(new, 8.12), expected_t812)

def test_param_specs_to_paths():
    #
    # If the time bases are not the same size, raise an error.
    #
    time_base0 = make_time_base(0.0, 10.0, 0.1)
    time_base1 = make_time_base(0.0, 10.0, 0.2)

    sim_inputs = [
        SimInput(time=time_base0, initial_state=None, params=None),
        SimInput(time=time_base1, initial_state=None, params=None),
    ]
    with pytest.raises(ValueError):
        param_specs_to_paths(sim_inputs)

    # 
    # Check that if the time bases are the same size but different values, "param_specs_to_paths" works.
    #
    time_base0 = make_time_base(0.0, 10.0, 0.1)
    time_base1 = make_time_base(0.0, 1.0, 0.01)
    sim_inputs = [
        SimInput(time=time_base0, initial_state=None, params=None),
        SimInput(time=time_base1, initial_state=None, params=None),
    ]
    assert len(param_specs_to_paths(sim_inputs)) == 2

    #
    # Check that if a non-monotonic time base is provided, an error is raised.
    #
    time_base =np.array([0.0, 1.0, 0.5])
    sim_inputs = [SimInput(time=time_base, initial_state=None, params={"a": {0.0: 0.1, 1.0: 0.2, 0.5: 0.3}})]
    with pytest.raises(ValueError):
        param_specs_to_paths(sim_inputs)

    #
    # Test dictionary resolution works.
    #
    simple_params = SimpleParams(a=1.0, b=2.0, c={0.0: 3.0, 1.0: 4.0, 2.0: -10.0})
    time_base = make_time_base(0.0, 2.0, 0.01)
    sim_input = SimInput(time=time_base, initial_state=None, params=simple_params)
    sim_input = param_specs_to_paths([sim_input])[0]
    sim_input_time_0 = pinterp.resolve_paths(sim_input, 0.0)
    sim_input_time_1 = pinterp.resolve_paths(sim_input, 1.0)
    sim_input_time_2 = pinterp.resolve_paths(sim_input, 2.0)
    assert sim_input_time_0.params.c == 3.0
    assert sim_input_time_1.params.c == 4.0
    assert sim_input_time_2.params.c == -10.0


    #
    # Test that we can provide an interpolation for a parameter.
    #

    simple_params = SimpleParams(a=1.0, b=2.0, c=pinterp.interp(time_base, 10.0 * time_base))
    sim_input = SimInput(time=time_base, initial_state=None, params=simple_params)
    sim_input = param_specs_to_paths([sim_input])[0]
    sim_input_time_0 = pinterp.resolve_paths(sim_input, 0.0)
    sim_input_time_1 = pinterp.resolve_paths(sim_input, 1.0)
    sim_input_time_2 = pinterp.resolve_paths(sim_input, 2.0)
    assert sim_input_time_0.params.c == 0.0
    assert sim_input_time_1.params.c == 10.0
    assert sim_input_time_2.params.c == 20.0