import os
import pytest

from popsim.ml.launch import launch_train

from popsim import DATA_DIR
from popsim.modules.transport_predictor.train_configs import BASE_CONFIG, update_submodule_configs

# Ensure each submodule from the transport predictor can be trained independently using all the default configs

@pytest.fixture(scope="session")
def transport_predictor_config():
    # Point the base config to the sample transport dataset
    transport_predictor_config = BASE_CONFIG.model_copy(
        update={
            "dataloader_config": {
                **BASE_CONFIG.dataloader_config,
                "ds_path": f"{DATA_DIR}/tcv/scrambled_transport_sample.nc",
            }
        }
    )

    # Update all the submodule configs to use the same dataloader config as the main module
    transport_predictor_config = update_submodule_configs(
        transport_predictor_config.model_dump(),
        [
            "profile_predictor",
            "power_balance",
            "p_oh_predictor",
            "p_rad_predictor",
        ]
    )

    return transport_predictor_config

@pytest.mark.parametrize("submodule", ["profile_predictor", "power_balance", "p_oh_predictor", "p_rad_predictor"])
def test_submodule_training(submodule, transport_predictor_config):
    submodule_config = transport_predictor_config.model_init_config["submodules"][submodule]

    submodule_config["max_epochs"] = 4 # Reduce epochs for testing purposes
    submodule_config["epochs_per_val"] = 2

    # Ensure the submodule config's dataloader_config matches the main module's dataloader_config except for the relevant keys
    for key in transport_predictor_config.dataloader_config.keys():
        if key not in ["state_vars", "input_vars", "target_vars", "extra_vars"]:
            assert submodule_config["dataloader_config"][key] == transport_predictor_config.dataloader_config[key]

    trainer, _, _, test_dl, _ = launch_train(submodule_config, use_wandb=False)
    assert trainer is not None
    assert test_dl is not None
