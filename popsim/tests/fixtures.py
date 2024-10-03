import xarray as xr
import os
import pytest
import popsim
import jax
import jax.numpy as jnp
import diffrax
from popsim.xarray_utils import solution_to_xarray
from functools import lru_cache

@lru_cache(maxsize=1)  # Caches only one result since it's always the same dataset
def load_cmod_test_dataset():
    try:
        path_to_data = os.path.join(popsim.DATA_DIR, "cmod/saperstein_july_2024_subset.nc")
        ds = xr.open_dataset(path_to_data)
        return ds
    except:
        raise ValueError(f"Could not find the CMOD test dataset at {path_to_data}. Did you do a git lfs init followed by a git lfs pull?")

def load_mast_thomson_test_dataset():
    path_to_data = os.path.join(popsim.DATA_DIR, "mast/mast_thomson_small_sample.nc")
    ds = xr.open_dataset(path_to_data)
    return ds

def generate_oscillator_dataset():
    """Generate an oscillator dataset, following the example from: https://docs.kidger.site/diffrax/examples/neural_ode """
    ts = jnp.linspace(0, 5, 50)
    cases = 64

    def generate_solution(key):
        y0 = jax.random.uniform(key, (2,), minval=-0.6, maxval=1)
        y0 = {"y0": y0[0], "y1": y0[1]}

        def f(t, y, args):
            y = jnp.array([y["y0"], y["y1"]])
            x = y / (1 + y)
            return {"y0": x[1], "y1": -x[0]}

        solver = diffrax.Tsit5()
        dt0 = 0.1
        saveat = diffrax.SaveAt(ts=ts)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
        )
        return sol
    keys = jax.random.split(jax.random.PRNGKey(0), cases)
    sols = jax.vmap(lambda key: generate_solution(key))(keys)
    return solution_to_xarray(sols, multi_simulation=True)

@pytest.fixture(scope="session")
def cmod_test_dataset():
    return load_cmod_test_dataset()

@pytest.fixture(scope="session")
def mast_thomson_test_dataset():
    return load_mast_thomson_test_dataset()

@pytest.fixture(scope="session")
def oscillator_dataset():
    return generate_oscillator_dataset()