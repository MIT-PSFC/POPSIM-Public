import typing

import equinox as eqx

from popsim.ml.envs import ModuleTrainingEnv

# Define models that are considered trainable.
TrainableModel = typing.Union[eqx.Module, ModuleTrainingEnv]
