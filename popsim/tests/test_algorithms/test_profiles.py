import jax
import jax.numpy as jnp

from popsim.algorithms.profiles import ProfileCalculator
from popsim.enums import Impurity
from popsim.tests import load_sparc_prd_data

def test_profile(load_sparc_prd_data):
    input_parameters, algorithm, points = load_sparc_prd_data
    calc = ProfileCalculator(n_points=100)
    import pdb; pdb.set_trace()