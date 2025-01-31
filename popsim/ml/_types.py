import typing
from dataclasses import dataclass

import equinox as eqx

from popsim.ml.envs import ModuleTrainingEnv

# Define models that are considered trainable.
TrainableModel = typing.Union[eqx.Module, ModuleTrainingEnv]


@dataclass
class TrainingMetadata:
    """Metadata for a training task."""

    @dataclass
    class TimeDepMetadata:
        state_init_vars: list[str]
        time_coord: str
        time_dim: str

    sample_coord: str
    sample_dim: str
    input_vars: list[str]
    target_vars: list[str]
    episode_coord: str
    episode_dim: str
    time_dep_metadata: typing.Optional[TimeDepMetadata] = None

    @property
    def is_time_dependent(self) -> bool:
        """Check if the metadata is for a time-dependent training task."""
        return self.time_dep_metadata is not None
