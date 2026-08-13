import pytest

import popsim.ml.envs as envs
from popsim.modules.module_examples import HybridExample


def test_eval_env_specification():
    class GoodEvalEnv(envs.ModuleEvalEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, observations, inputs):
            pass

        def create_inputs(self, inputs):
            pass

    module = HybridExample()
    eval_env = GoodEvalEnv(module)
    assert eval_env is not None

    class EvalEnvWrongName(envs.ModuleEvalEnv):
        modules: HybridExample

        def __init__(self, module: HybridExample):
            self.modules = module

        def create_state(self, observations, inputs):
            pass

        def create_inputs(self, inputs):
            pass

    with pytest.raises(TypeError):
        eval_env = EvalEnvWrongName(module)

    class EvalEnvMissingMethod(envs.ModuleEvalEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, observations, inputs):
            pass

    with pytest.raises(TypeError):
        eval_env = EvalEnvMissingMethod(module)


def test_train_env_specification():
    module = HybridExample()

    class GoodTrainEnv(envs.ModuleTrainingEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, observations, inputs):
            pass

        def create_inputs(self, inputs):
            pass

        def get_trainable(self):
            pass

    train_env = GoodTrainEnv(module)
    assert train_env is not None

    class TrainEnvMissingMethod(envs.ModuleTrainingEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, observations, inputs):
            pass

        def create_inputs(self, inputs):
            pass

    with pytest.raises(TypeError):
        train_env = TrainEnvMissingMethod(module)
