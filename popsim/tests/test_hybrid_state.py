import pytest
from popsim import hybrid_state
import chex
import jax

@pytest.fixture
def state_classes():
    @chex.dataclass
    class SubState:
        discrete_foo: int = hybrid_state.discrete_time_field(0)
        cont_bar: float = 2.0

    @chex.dataclass
    class State:
        example_state: int = hybrid_state.discrete_time_field(0)
        example_cont: float = 0.0
        sub_state: SubState = SubState()

    return State, SubState

@pytest.fixture
def state_instance(state_classes):
    State, _ = state_classes
    return State()


def test_create_discrete_state_filter_spec(state_classes, state_instance):
    State, SubState = state_classes
    s = state_instance

    expected = State(example_state=True, example_cont=False, sub_state=SubState(discrete_foo=True, cont_bar=False))

    result = hybrid_state.create_discrete_state_filter_spec(s)

    chex.assert_trees_all_equal(result, expected)

def test_hybrid_state(state_classes, state_instance):
    State, SubState = state_classes
    s = state_instance

    expected_discrete = State(example_state=s.example_state, example_cont=None, sub_state=SubState(discrete_foo=s.sub_state.discrete_foo, cont_bar=None))
    expected_continuous = State(example_state=None, example_cont=s.example_cont, sub_state=SubState(discrete_foo=None, cont_bar=s.sub_state.cont_bar))

    discretes, conts = jax.jit(hybrid_state.partition_discrete_cont)(s)

    chex.assert_trees_all_equal(discretes, expected_discrete)
    chex.assert_trees_all_equal(conts, expected_continuous)
