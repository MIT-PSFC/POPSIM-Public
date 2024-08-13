import typing

import chex
import diffrax
import jax.tree_util as tu
from jaxtyping import Array, ArrayLike, Float, PyTree

"""
A "PathSpec" is a user-specification for a time dependent trajectory.
The keys are "time", and the values are a sequence of PyTrees (e.g. dictionaries).
For example, one valid PathSpec is:
    {0.0: -10, 1.0: -5.0}
Example of a case where it's a PyTree:
    {0.0: {"x": 1.0, "y": 2.0}, 1.0: {"x": 2.0, "y": 3.0}}
Array elements are also valid:
    {0.0: [1.0, 2.0], 1.0: [2.0, 3.0]}
"""
PathSpec = dict[Float[ArrayLike, ""], PyTree[ArrayLike]]

# A PathSpec or a Trajectory.
PathOrPathSpec = diffrax.AbstractPath | PathSpec

# A constant or a time dependent value.
# This type is the user-facing API for specifying parameters.
ConstantOrPathSpec = typing.Any | PathOrPathSpec

# Once the user-facing API is parsed, params take the form of a PyTree of ConstantOrPath.
ConstantOrPath = typing.Any | diffrax.AbstractPath

# A user-specification for a params tree.
ParamSpec = PyTree[ConstantOrPathSpec]

# Type alias for a Jax PyTree key.
PyTreeKey = typing.Union[tu.SequenceKey, tu.DictKey, tu.GetAttrKey]


@chex.dataclass
class SimulationInput:
    ts: Array
    initial_state: PyTree
    params: PyTree
