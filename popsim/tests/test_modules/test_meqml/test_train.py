from popsim.modules.meqml.fast_obs.config import FAST_OBS_CONFIG
from popsim.modules.meqml.liuqe.config import LIUQE_CONFIG
from popsim.modules.meqml.fgs.config import FGS_CONFIG
from popsim.modules.meqml.fbt.config import FBT_SURROGATE_CONFIG
import copy
from popsim.ml.launch import launch_train
from popsim.data.tcv import SCRAMBLED_LIUQE_PATH, SCRAMBLED_FBT_PATH

"""
Test that the training loops can run without error on scrambled data.
"""
def test_train_fast_obs():
    config = copy.deepcopy(FAST_OBS_CONFIG)
    config["dataloader_config"]["path"] = SCRAMBLED_LIUQE_PATH
    config["max_epochs"] = 2
    config["epochs_per_val"] = 1
    launch_train(config)
    
def test_train_liuqe():
    config = copy.deepcopy(LIUQE_CONFIG)
    config["dataloader_config"]["path"] = SCRAMBLED_LIUQE_PATH
    config["max_epochs"] = 2
    config["epochs_per_val"] = 1
    launch_train(config)

def test_train_fgs():
    config = copy.deepcopy(FGS_CONFIG)
    config["dataloader_config"]["path"] = SCRAMBLED_LIUQE_PATH
    config["max_epochs"] = 2
    config["epochs_per_val"] = 1
    launch_train(config)

def test_train_fbt():
    config = copy.deepcopy(FBT_SURROGATE_CONFIG)
    config["dataloader_config"]["path"] = SCRAMBLED_FBT_PATH
    config["max_epochs"] = 2
    config["epochs_per_val"] = 1
    launch_train(config)