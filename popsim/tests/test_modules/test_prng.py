from popsim.modules.prng import PRNGModule
import jax.numpy as jnp
from popsim.simulate import simulate
from popsim.param_utils import make_time_base

def test_prng_module():
    module = PRNGModule()
    state = PRNGModule.State(seed=0)
    params = PRNGModule.Params()
    time_base = make_time_base(0.0, 1.0, 1e-3)

    sol_xarray = simulate(module, time_base, state, params)

    seeds = sol_xarray["state.seed"].values
    
    # Check that we have the correct number of seeds.
    assert seeds.shape == time_base.shape

    # Check that all the seed values are unique.
    assert jnp.any(seeds[0] == seeds[1:]) == False