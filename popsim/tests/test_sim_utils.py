from popsim.sim_utils import SimInput, make_time_base, MultiCases, CombinatorialCases, StaticSampler, draw_sample, sample_samplers
import chex
import pytest
import jax
import jax.numpy as jnp
import numpy as np

def test_generate_cases():
    @chex.dataclass
    class Data:
        a: int
        b: int
        c: int
    @chex.dataclass
    class ExampleState:
        data: Data
        x: float

    @chex.dataclass
    class ExampleParams:
        p0: float
        p1: dict[str, float]

    time_base = make_time_base(0.0, 1.0, 0.01)
    initial_state = ExampleState(data=Data(a=1, b=2, c=3), x=0.0)
    params = ExampleParams(p0=1.0, p1={"a": 3.0, "b": 4.0})
    
    sim_input = SimInput(time=time_base, initial_state=initial_state, params=params)

    #
    # Check the identity case.
    #
    cases = sim_input.generate_sim_cases()
    assert isinstance(cases, SimInput)


    #
    # Try using MultiCases for a single subtree.
    #
    initial_state_multi = ExampleState(data=MultiCases(cases=[Data(a=1, b=2, c=3), Data(a=-1, b=-2, c=-3)]), x=0.0)

    sim_input = SimInput(time=time_base, initial_state=initial_state_multi, params=params)
    cases = sim_input.generate_sim_cases()
    assert len(cases) == 2
    assert cases[0] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=1, b=2, c=3), x=0.0), params=params)
    assert cases[1] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=-1, b=-2, c=-3), x=0.0), params=params)

    #
    # Try using MultiCases for both varying state and params.
    #
    params_multi = ExampleParams(p0=MultiCases(cases=[1.0, 2.0]), p1=MultiCases(cases=[{"a": 3.0, "b": 4.0}, {"a": -3.0, "b": -4.0}]))
    sim_input = SimInput(time=time_base, initial_state=initial_state_multi, params=params_multi)
    cases = sim_input.generate_sim_cases()
    assert len(cases) == 2
    assert cases[0] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=1, b=2, c=3), x=0.0), params=ExampleParams(p0=1.0, p1={"a": 3.0, "b": 4.0}))
    assert cases[1] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=-1, b=-2, c=-3), x=0.0), params=ExampleParams(p0=2.0, p1={"a": -3.0, "b": -4.0}))

    #
    # Check that an error is raised if MultiCases don't have the same length.
    #
    params_multi = ExampleParams(p0=MultiCases(cases=[1.0, 2.0, 3.0]), p1=MultiCases(cases=[{"a": 3.0, "b": 4.0}, {"a": -3.0, "b": -4.0}, {"a": 3.0, "b": 4.0}]))
    sim_input = SimInput(time=time_base, initial_state=initial_state_multi, params=params_multi)
    with pytest.raises(ValueError):
        cases = sim_input.generate_sim_cases()

    #
    # Check that an error is raised if both MultiCases and CombinatorialCases are present. 
    #
    params_multi = ExampleParams(p0=MultiCases(cases=[1.0, 2.0]), p1=CombinatorialCases(cases=[{"a": 3.0, "b": 4.0}, {"a": -3.0, "b": -4.0}]))
    sim_input = SimInput(time=time_base, initial_state=initial_state_multi, params=params_multi)
    with pytest.raises(ValueError):
        cases = sim_input.generate_sim_cases()


    #
    # Try using CombintorialCases for a single subtree.
    #
    initial_state_comb = ExampleState(data=CombinatorialCases(cases=[Data(a=1, b=2, c=3), Data(a=-1, b=-2, c=-3)]), x=0.0)
    sim_input = SimInput(time=time_base, initial_state=initial_state_comb, params=params)
    cases = sim_input.generate_sim_cases()
    assert len(cases) == 2
    assert cases[0] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=1, b=2, c=3), x=0.0), params=params)
    assert cases[1] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=-1, b=-2, c=-3), x=0.0), params=params)


    #
    # Try using CombinatorialCases with one parameter having 2 cases and the other having 3 cases. Expect 6 cases.
    #
    initial_state_comb = ExampleState(data=CombinatorialCases(cases=[Data(a=1, b=2, c=3), Data(a=-1, b=-2, c=-3)]), x=0.0)
    params_comb = ExampleParams(p0=CombinatorialCases(cases=[1.0, 2.0, 3.0]), p1={"a": 3.0, "b": 4.0})
    sim_input = SimInput(time=time_base, initial_state=initial_state_comb, params=params_comb)
    cases = sim_input.generate_sim_cases()
    assert len(cases) == 6
    assert cases[0] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=1, b=2, c=3), x=0.0), params=ExampleParams(p0=1.0, p1={"a": 3.0, "b": 4.0}))
    assert cases[1] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=1, b=2, c=3), x=0.0), params=ExampleParams(p0=2.0, p1={"a": 3.0, "b": 4.0}))
    assert cases[2] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=1, b=2, c=3), x=0.0), params=ExampleParams(p0=3.0, p1={"a": 3.0, "b": 4.0}))
    assert cases[3] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=-1, b=-2, c=-3), x=0.0), params=ExampleParams(p0=1.0, p1={"a": 3.0, "b": 4.0}))
    assert cases[4] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=-1, b=-2, c=-3), x=0.0), params=ExampleParams(p0=2.0, p1={"a": 3.0, "b": 4.0}))
    assert cases[5] == SimInput(time=time_base, initial_state=ExampleState(data=Data(a=-1, b=-2, c=-3), x=0.0), params=ExampleParams(p0=3.0, p1={"a": 3.0, "b": 4.0}))

def test_static_sampler():
    def sample_fn(key):
        return jax.random.uniform(key, (3,))

    sampler = StaticSampler(sample_fn)
    assert isinstance(sampler, StaticSampler)
    key = jax.random.PRNGKey(0)
    assert sampler(key).shape == (3,)

    def sampler_fn_np(key):
        np.random.seed(key[0])
        return np.random.uniform(size=(3,))
    
    sampler = StaticSampler(sampler_fn_np)
    assert isinstance(sampler, StaticSampler)
    key = np.array([0])
    assert sampler(key).shape == (3,)


def test_sample_sim_input():
    def sample_fn(key):
        return jax.random.uniform(key, (3,))
    tree = {
        "random": StaticSampler(sample_fn),
        "not_random": 1.0,
    }

    sim_input = SimInput(time=np.array([0.0, 1.0]), initial_state=tree, params=tree)

    samples = sim_input.sample(jax.random.PRNGKey(0), 3) # Sample 3 times.

    # Check not_random is the same in all samples.
    assert all([sample.initial_state["not_random"] == 1.0 for sample in samples])
    assert all([sample.params["not_random"] == 1.0 for sample in samples])

    # Check that random is different in all samples.
    randoms1 = [sample.initial_state["random"] for sample in samples]
    assert jnp.not_equal(randoms1[0], randoms1[1]).all()
    assert jnp.not_equal(randoms1[0], randoms1[2]).all()
    assert jnp.not_equal(randoms1[1], randoms1[2]).all()
    randoms2 = [sample.params["random"] for sample in samples]
    assert jnp.not_equal(randoms2[0], randoms2[1]).all()
    assert jnp.not_equal(randoms2[0], randoms2[2]).all()
    assert jnp.not_equal(randoms2[1], randoms2[2]).all()


    assert len(samples) == 3


def test_sample_samplers():
    def sample_fn(key):
        return jax.random.uniform(key, (3,))
    tree = {
        "random": StaticSampler(sample_fn),
        "not_random": 1.0,
    }
    out = draw_sample(jax.random.PRNGKey(0), tree)
    out["random"].shape == (3,)
    out["not_random"] == 1.0

    samples = sample_samplers(tree, jax.random.PRNGKey(0), 3)
    assert len(samples) == 3
    assert all([sample["not_random"] == 1.0 for sample in samples])
    randoms = [sample["random"] for sample in samples]
    assert jnp.not_equal(randoms[0], randoms[1]).all()
    assert jnp.not_equal(randoms[0], randoms[2]).all()
    assert jnp.not_equal(randoms[1], randoms[2]).all()
