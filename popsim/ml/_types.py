import typing

import equinox as eqx

from popsim.ml.envs import ModuleEvalEnv

# Define models that are considered trainable.
TrainableModel = typing.Union[eqx.Module, ModuleEvalEnv]
