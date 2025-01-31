import popsim.ml.envs as envs
from popsim.modules.module_examples import HybridExample
import pytest

def test_eval_env_specification():
    class GoodEvalEnv(envs.ModuleEvalEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, input_dict):
            pass

        def create_inputs(self, input_dict):
            pass
        

    module = HybridExample()
    eval_env = GoodEvalEnv(module)
    assert eval_env is not None

    class EvalEnvWrongName(envs.ModuleEvalEnv):
        modules: HybridExample

        def __init__(self, module: HybridExample):
            self.modules = module

        def create_state(self, input_dict):
            pass

        def create_inputs(self, input_dict):
            pass
    
    with pytest.raises(ValueError):
        eval_env = EvalEnvWrongName(module)
    
    class EvalEnvMissingMethod(envs.ModuleEvalEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, input_dict):
            pass
        
    
    with pytest.raises(TypeError):
        eval_env = EvalEnvMissingMethod(module)

def test_train_env_specification():
    module = HybridExample()


    class GoodTrainEnv(envs.ModuleTrainingEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, input_dict):
            pass

        def create_inputs(self, input_dict):
            pass
        
        def get_trainable(self):
            pass
        
    train_env = GoodTrainEnv(module)
    assert train_env is not None

    
    class TrainEnvMissingMethod(envs.ModuleTrainingEnv):
        module: HybridExample

        def __init__(self, module: HybridExample):
            self.module = module

        def create_state(self, input_dict):
            pass

        def create_inputs(self, input_dict):
            pass
        
    
    with pytest.raises(TypeError):
        train_env = TrainEnvMissingMethod(module)