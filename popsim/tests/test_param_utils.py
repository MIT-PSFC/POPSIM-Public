import dataclasses
import jax
import chex
import pytest
from jaxtyping import ArrayLike
import numpy as np

import popsim.enums as penums
import popsim.interp as pinterp
from popsim.simulate import make_time_base, SimInput
from popsim.input_utils import (
    build_input_paths,
    input_specs_to_paths
)


@chex.dataclass
class SimpleInputs:
    a: float
    b: float
    c: float


# Note: cubic case currently breaks as extrapolation of the time dictionary is not handeled as expected.
@pytest.mark.parametrize("interp_type", [pinterp.InterpType.LINEAR])
def test_build_input_paths(interp_type):
    # We want to ensure that the built input paths match exactly,
    # but assert_trees_all_equal fails for float32 mismatches against python float literals.
    if jax.config.jax_enable_x64:
        assert_trees_match = chex.assert_trees_all_equal
    else:
        assert_trees_match = chex.assert_trees_all_close

    @chex.dataclass
    class Inputs:
        a: float
        nested_b: dict[str, ArrayLike]
        imps: dict[penums.Impurity, float]

    inputs = Inputs(
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
    new_inputs = build_input_paths(inputs, time_base, interp_type)
    assert_trees_match(pinterp.resolve_paths(new_inputs, time_base[0]), inputs)
    assert_trees_match(pinterp.resolve_paths(new_inputs, time_base[-1]), inputs)

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
    inputs.imps = time_dep_imps
    new_inputs2 = build_input_paths(inputs, time_base, interp_type)
    interped_imps = pinterp.interp_time_dic(time_dep_imps, interp_type)
    expected_inputs_begin = Inputs(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=time_dep_imps[0.0])
    expected_inputs_t25 = Inputs(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps=time_dep_imps[2.5])

    assert_trees_match(pinterp.resolve_paths(new_inputs2, 0.0), expected_inputs_begin)
    assert_trees_match(pinterp.resolve_paths(new_inputs2, 2.5), expected_inputs_t25)


    #
    # Now have the user specify a time-dependent impurity sequence but only for Tungsten.
    #
    inputs.imps = {
        penums.Impurity.Tungsten: {0.0: 4.0, 1.0: 5.0, 2.5: 6.0},
        penums.Impurity.Neon: 5.0,
    }
    new_inputs3 = build_input_paths(inputs, time_base, interp_type)
    expected_inputs3_begin = Inputs(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 4.0, penums.Impurity.Neon: 5.0})
    expected_inputs_3_t25 = Inputs(a=1.0, nested_b={"b0": 2.0, "b1": [3.0, -3.0]}, imps={penums.Impurity.Tungsten: 6.0, penums.Impurity.Neon: 5.0})
    assert_trees_match(pinterp.resolve_paths(new_inputs3, 0.0), expected_inputs3_begin)
    assert_trees_match(pinterp.resolve_paths(new_inputs3, 2.5), expected_inputs_3_t25)


    #
    # Check that we can specify a time-dependent subtree.
    #
    time_dep_b = {
        0.0: {"b0": 2.0, "b1": [3.0, -3.0]},
        1.0: {"b0": 3.0, "b1": [4.0, -4.0]},
        2.5: {"b0": 4.0, "b1": [5.0, -5.0]},
    }
    time_dep_inputs = Inputs(
        a=1.0,
        nested_b=time_dep_b,
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )
    built_inputs = build_input_paths(time_dep_inputs, time_base, interp_type)

    b_at_zero = pinterp.resolve_paths(built_inputs.nested_b, 0.0)
    b_at_point_five = pinterp.resolve_paths(built_inputs.nested_b, 0.5)
    b_at_one = pinterp.resolve_paths(built_inputs.nested_b, 1.0)
    b_at_25 = pinterp.resolve_paths(built_inputs.nested_b, 2.5)

    if interp_type == "linear":
        expected_at_point_five = {"b0": 2.5, "b1": [3.5, -3.5]}
        assert_trees_match(b_at_point_five, expected_at_point_five)

    assert_trees_match(b_at_zero, time_dep_b[0.0])
    assert_trees_match(b_at_one, time_dep_b[1.0])
    assert_trees_match(b_at_25, time_dep_b[2.5])


    #
    # Check the case where a inputs is an interp.
    #
    interped_a = pinterp.interp(time_base, time_base, interp_type=interp_type)
    inputs = Inputs(
        a=interped_a,
        nested_b={"b0": 2.0, "b1": [3.0, -3.0]},
        imps={
            penums.Impurity.Tungsten: 4.0,
            penums.Impurity.Neon: 5.0,
        },
    )
    new = build_input_paths(inputs, time_base, interp_type)

    expected_t0 = dataclasses.replace(inputs, a=0.0)
    expected_t15 = dataclasses.replace(inputs, a=1.5)
    expected_t812 = dataclasses.replace(inputs, a=8.12)
    assert_trees_match(pinterp.resolve_paths(new, 0.0), expected_t0)
    assert_trees_match(pinterp.resolve_paths(new, 1.5), expected_t15)
    assert_trees_match(pinterp.resolve_paths(new, 8.12), expected_t812)

def test_input_specs_to_paths():
    #
    # If the time bases are not the same size, raise an error.
    #
    time_base0 = make_time_base(0.0, 10.0, 0.1)
    time_base1 = make_time_base(0.0, 10.0, 0.2)

    sim_inputs = [
        SimInput(time=time_base0, initial_state=None, inputs=None),
        SimInput(time=time_base1, initial_state=None, inputs=None),
    ]
    with pytest.raises(ValueError):
        input_specs_to_paths(sim_inputs)

    # 
    # Check that if the time bases are the same size but different values, "input_specs_to_paths" works.
    #
    time_base0 = make_time_base(0.0, 10.0, 0.1)
    time_base1 = make_time_base(0.0, 1.0, 0.01)
    sim_inputs = [
        SimInput(time=time_base0, initial_state=None, inputs=None),
        SimInput(time=time_base1, initial_state=None, inputs=None),
    ]
    assert len(input_specs_to_paths(sim_inputs)) == 2

    #
    # Check that if a non-monotonic time base is provided, an error is raised.
    #
    time_base =np.array([0.0, 1.0, 0.5])
    sim_inputs = [SimInput(time=time_base, initial_state=None, inputs={"a": {0.0: 0.1, 1.0: 0.2, 0.5: 0.3}})]
    with pytest.raises(ValueError):
        input_specs_to_paths(sim_inputs)

    #
    # Test dictionary resolution works.
    #
    simple_inputs = SimpleInputs(a=1.0, b=2.0, c={0.0: 3.0, 1.0: 4.0, 2.0: -10.0})
    time_base = make_time_base(0.0, 2.0, 0.01)
    sim_input = SimInput(time=time_base, initial_state=None, inputs=simple_inputs)
    sim_input = input_specs_to_paths([sim_input])[0]
    sim_input_time_0 = pinterp.resolve_paths(sim_input, 0.0)
    sim_input_time_1 = pinterp.resolve_paths(sim_input, 1.0)
    sim_input_time_2 = pinterp.resolve_paths(sim_input, 2.0)
    assert sim_input_time_0.inputs.c == 3.0
    assert sim_input_time_1.inputs.c == 4.0
    assert sim_input_time_2.inputs.c == -10.0


    #
    # Test that we can provide an interpolation for an input.
    #

    simple_inputs = SimpleInputs(a=1.0, b=2.0, c=pinterp.interp(time_base, 10.0 * time_base))
    sim_input = SimInput(time=time_base, initial_state=None, inputs=simple_inputs)
    sim_input = input_specs_to_paths([sim_input])[0]
    sim_input_time_0 = pinterp.resolve_paths(sim_input, 0.0)
    sim_input_time_1 = pinterp.resolve_paths(sim_input, 1.0)
    sim_input_time_2 = pinterp.resolve_paths(sim_input, 2.0)
    assert sim_input_time_0.inputs.c == 0.0
    assert sim_input_time_1.inputs.c == 10.0
    assert sim_input_time_2.inputs.c == 20.0