import pytest
import chex
from dataclasses import field
import jax
from popsim.field_labels import (
    discrete_time_field,
    no_save_field,
    discrete_no_save_field,
    create_filter_spec,
    partition_discrete_cont,
    partition_save_no_save,
)


@pytest.fixture
def state_classes():
    """Fixture for creating example state classes with discrete, continuous, and no-save fields."""

    @chex.dataclass
    class SubState:
        discrete_foo: int = discrete_time_field(default=0)
        cont_bar: float = 2.0
        discrete_no_save_field: int = discrete_no_save_field(default=0)
        no_save_field: int = no_save_field(default=0)

    @chex.dataclass
    class State:
        example_state: int = discrete_time_field(default=0)
        example_cont: float = 0.0
        no_save_example: int = no_save_field(default=0)
        discrete_no_save_example: int = discrete_no_save_field(default=0)
        sub_state: SubState = field(default_factory=SubState)

    return State, SubState


@pytest.fixture
def state_instance(state_classes):
    """Fixture for creating an instance of the State class."""
    State, _ = state_classes
    return State()


def test_create_discrete_state_filter_spec(state_classes, state_instance):
    """
    Test that create_filter_spec correctly labels the fields of a dataclass that are discrete time states.
    """
    State, SubState = state_classes
    s = state_instance

    # Expected structure with discrete fields marked as True
    expected = State(
        example_state=True,
        example_cont=False,
        no_save_example=False,
        discrete_no_save_example=True,
        sub_state=SubState(
            discrete_foo=True,
            cont_bar=False,
            discrete_no_save_field=True,
            no_save_field=False,
        ),
    )

    # Generate filter spec with discrete time condition
    result = create_filter_spec(s, lambda f: f.metadata.get("discrete_state", False))

    chex.assert_trees_all_equal(result, expected)


def test_partition_discrete_cont(state_classes, state_instance):
    """
    Test that partition_discrete_cont correctly partitions the state into discrete and continuous components.
    """
    State, SubState = state_classes
    s = state_instance

    # Expected discrete and continuous partitions
    expected_discrete = State(
        example_state=s.example_state,
        example_cont=None,
        no_save_example=None,
        discrete_no_save_example=s.discrete_no_save_example,
        sub_state=SubState(
            discrete_foo=s.sub_state.discrete_foo,
            cont_bar=None,
            discrete_no_save_field=s.sub_state.discrete_no_save_field,
            no_save_field=None,
        ),
    )
    expected_continuous = State(
        example_state=None,
        example_cont=s.example_cont,
        no_save_example=s.no_save_example,
        discrete_no_save_example=None,
        sub_state=SubState(
            discrete_foo=None,
            cont_bar=s.sub_state.cont_bar,
            discrete_no_save_field=None,
            no_save_field=s.sub_state.no_save_field,
        ),
    )

    # Partition the state
    discretes, conts = jax.jit(partition_discrete_cont)(s)

    chex.assert_trees_all_equal(discretes, expected_discrete)
    chex.assert_trees_all_equal(conts, expected_continuous)


def test_partition_save_no_save(state_classes, state_instance):
    """
    Test that partition_save_no_save correctly partitions the state into fields to save and not to save.
    """
    State, SubState = state_classes
    s = state_instance

    # Expected saved and not saved partitions
    expected_save = State(
        example_state=s.example_state,
        example_cont=s.example_cont,
        no_save_example=None,
        discrete_no_save_example=None,
        sub_state=SubState(
            discrete_foo=s.sub_state.discrete_foo,
            cont_bar=s.sub_state.cont_bar,
            discrete_no_save_field=None,
            no_save_field=None,
        ),
    )
    expected_no_save = State(
        example_state=None,
        example_cont=None,
        no_save_example=s.no_save_example,
        discrete_no_save_example=s.discrete_no_save_example,
        sub_state=SubState(
            discrete_foo=None,
            cont_bar=None,
            discrete_no_save_field=s.sub_state.discrete_no_save_field,
            no_save_field=s.sub_state.no_save_field,
        ),
    )

    # Partition the state based on no_save fields
    saves, no_saves = jax.jit(partition_save_no_save)(s)

    chex.assert_trees_all_equal(saves, expected_save)
    chex.assert_trees_all_equal(no_saves, expected_no_save)


def test_discrete_no_save_field(state_classes, state_instance):
    """
    Test that fields marked with discrete_no_save_field are correctly identified as both discrete and no_save.
    """
    State, SubState = state_classes
    s = state_instance

    # Generate filter spec for discrete_state
    discrete_spec = create_filter_spec(s, lambda f: f.metadata.get("discrete_state", False))
    # Generate filter spec for no_save
    no_save_spec = create_filter_spec(s, lambda f: f.metadata.get("no_save", False))

    # Check that the discrete_no_save_example and discrete_no_save_field are marked in both specs
    assert getattr(discrete_spec, 'discrete_no_save_example') is True
    assert getattr(no_save_spec, 'discrete_no_save_example') is True

    assert getattr(discrete_spec.sub_state, 'discrete_no_save_field') is True
    assert getattr(no_save_spec.sub_state, 'discrete_no_save_field') is True

    # Now partition the state using both partition functions
    discretes, _ = jax.jit(partition_discrete_cont)(s)
    saves, no_saves = jax.jit(partition_save_no_save)(s)

    # The discrete_no_save_example should be in both discretes and no_saves
    assert getattr(discretes, 'discrete_no_save_example') == s.discrete_no_save_example
    assert getattr(no_saves, 'discrete_no_save_example') == s.discrete_no_save_example

    # The field should be None in the continuous and saves partitions
    _, conts = jax.jit(partition_discrete_cont)(s)
    assert getattr(conts, 'discrete_no_save_example') is None
    assert getattr(saves, 'discrete_no_save_example') is None

    # Repeat for sub_state.discrete_no_save_field
    assert getattr(discretes.sub_state, 'discrete_no_save_field') == s.sub_state.discrete_no_save_field
    assert getattr(no_saves.sub_state, 'discrete_no_save_field') == s.sub_state.discrete_no_save_field
    assert getattr(conts.sub_state, 'discrete_no_save_field') is None
    assert getattr(saves.sub_state, 'discrete_no_save_field') is None
