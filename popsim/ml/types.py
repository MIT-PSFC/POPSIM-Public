import chex
from jaxtyping import Array

from popsim.simulate import SimInput


@chex.dataclass
class EvalInput:
    """Input data structure for evaluating a module."""

    sim_input: SimInput  # Standard input data structure for running a simulation.
    targets: dict[str, Array]  # Dict of target variables to compare module outputs against.
