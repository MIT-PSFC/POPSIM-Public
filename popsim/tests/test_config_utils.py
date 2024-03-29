import chex
from popsim.config_utils import CombinatorialCases, generate_combinations

def test_generate_params_combinations():
    @chex.dataclass
    class Params:
        a: float
        b: float
        c: float


    # Example usage
    params = Params(a=CombinatorialCases([1.0, 2.0, 3.0]), b=2.0, c=CombinatorialCases([3.0, 4.0]))
    combinations = generate_combinations(params)

    assert len(combinations) == 6
    assert combinations[0] == Params(a=1.0, b=2.0, c=3.0)
    assert combinations[1] == Params(a=1.0, b=2.0, c=4.0)
    assert combinations[2] == Params(a=2.0, b=2.0, c=3.0)
    assert combinations[3] == Params(a=2.0, b=2.0, c=4.0)
    assert combinations[4] == Params(a=3.0, b=2.0, c=3.0)
    assert combinations[5] == Params(a=3.0, b=2.0, c=4.0)
