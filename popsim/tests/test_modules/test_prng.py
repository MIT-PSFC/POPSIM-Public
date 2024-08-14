from popsim.modules.prng import PRNGModule
import jax.numpy as jnp
from popsim.simulate import simulate, make_time_base, SimInput

def test_prng_module():
    module = PRNGModule()
    state = PRNGModule.State(seed=0)
    params = PRNGModule.Params()
    time_base = make_time_base(0.0, 1.0, 1e-3)

    sol_xarray = simulate(module, SimInput(time=time_base, initial_state=state, params=params))

    seeds = sol_xarray["state.seed"].values
    
    # Check that we have the correct number of seeds.
    assert seeds.shape == time_base.shape

    # Check that all the seed values are unique.
    assert jnp.any(seeds[0] == seeds[1:]) == False


    # Test that every time we initialize PRNGModule.State without a seed, we get a different seed.
    states = [PRNGModule.State() for _ in range(100)]
    seeds = [state.seed for state in states]
    assert len(seeds) == len(set(seeds))