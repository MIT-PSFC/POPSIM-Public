from abc import ABC, abstractmethod

import chex
from jaxtyping import ArrayLike

from popsim import ModuleBase


@chex.dataclass
class ModuleEvalEnv(ABC):
    module: ModuleBase

    def __init__(self, module: ModuleBase):
        self.module = module

    @staticmethod
    @abstractmethod
    def create_state(data: dict[str, ArrayLike]) -> "State":  # noqa: F821
        pass

    @staticmethod
    @abstractmethod
    def create_params(data: dict[str, ArrayLike]) -> "Params":  # noqa: F821
        pass


@chex.dataclass
class ModuleTrainingEnv(ABC):
    module_eval_env: ModuleEvalEnv

    def __init__(self, module_eval_env: ModuleEvalEnv):
        self.module_eval_env = module_eval_env

    @abstractmethod
    def get_trainable_values(self) -> tuple[object]:
        pass

    @staticmethod
    @abstractmethod
    def loss(sim_data: dict[str, ArrayLike], data: dict[str, ArrayLike]) -> float:
        pass
