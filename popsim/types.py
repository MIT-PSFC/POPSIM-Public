import typing

import diffrax
from jaxtyping import ArrayLike, Float, PyTree

"""
A "TrajectorySpec" is a user-specification for a time dependent trajectory.
The keys are "time", and the values are a sequence of PyTrees (e.g. dictionaries).
For example, one valid TrajectorySpec is:
    {0.0: -10, 1.0: -5.0}
Example of a case where it's a PyTree:
    {0.0: {"x": 1.0, "y": 2.0}, 1.0: {"x": 2.0, "y": 3.0}}
Array elements are also valid:
    {0.0: [1.0, 2.0], 1.0: [2.0, 3.0]}
"""
TrajectorySpec = dict[Float[ArrayLike, ""], PyTree[ArrayLike]]

# A TrajectorySpec or a Trajectory.
TrajectoryOrTrajectorySpec = diffrax.AbstractPath | TrajectorySpec


# A constant or a time dependent value.
# This type is the user-facing API for config.
ConstantOrTimeDependentSpec = typing.Any | TrajectoryOrTrajectorySpec

# Once the user-facing API is parsed, configs take the form of a PyTree of ConstantOrTimeDependent.
ConstantOrTimeDependent = typing.Any | diffrax.AbstractPath
