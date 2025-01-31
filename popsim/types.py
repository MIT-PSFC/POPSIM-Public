import typing

import diffrax
import jax.tree_util as tu
from jax import Array
from jaxtyping import ArrayLike, Float, PRNGKeyArray, PyTree

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
# This type is the user-facing API for specifying inputs.
ConstantOrPathSpec = typing.Any | PathOrPathSpec

# Once the user-facing API is parsed, inputs take the form of a PyTree of ConstantOrPath.
ConstantOrPath = typing.Any | diffrax.AbstractPath

# A user-specification for a inputs tree.
Inputspec = PyTree[ConstantOrPathSpec]

# Type alias for a Jax PyTree key.
PyTreeKey = typing.Union[tu.SequenceKey, tu.DictKey, tu.GetAttrKey]


@typing.runtime_checkable
class StaticSamplerFn(typing.Protocol):
    def __call__(self, key: PRNGKeyArray) -> Array:
        ...
