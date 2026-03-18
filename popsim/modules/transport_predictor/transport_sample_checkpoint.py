import os
import shutil

import fire
import xarray as xr
from loguru import logger

from popsim import DATA_DIR, PACKAGE_ROOT
from popsim.data import get_path_to_ml_data_dump
from popsim.ml.launch import launch_train
from popsim.ml.train_config import TrainConfig
from popsim.ml.trainer import Trainer
from popsim.modules.transport_predictor.train_configs import BASE_CONFIG, update_submodule_configs
from popsim.modules.transport_predictor.training_run_builder import TransportPredictorTrainRunBuilder

CHECKPOINT_DIR_BASE = os.path.join(PACKAGE_ROOT, "checkpoints", "transport_predictor_demo")
MAX_EPOCHS = 800
EPOCHS_PER_VAL = 20


##################################
# Section 2.1: Set up the config #
##################################
def setup_main_config(dataset: str) -> TrainConfig:
    """Set up the main config for the TransportPredictor module and all its submodules.
    We are including three model checkpoints, trained on different datasets.

    "tcv_11" is meant to demonstrate the training pipeline. It's very small so we train/val/test on the same data. This is CHEATING, but fine for a quick demo.
    "tcv_132" is similar to the one used for the result published in https://arxiv.org/abs/2509.10244.

    Args:
        dataset (str): Which dataset to use for training. Must be one of "tcv_11", "tcv_132".

    Returns:
        TrainConfig: The main config for the TransportPredictor module and all its submodules, with the appropriate dataloader config for the specified dataset.
    """

    sample_ds_path = os.path.join(DATA_DIR, "tcv", "scrambled_transport_sample.nc")
    sample_ds = xr.open_dataset(sample_ds_path)
    test_shots = sample_ds.shot.values.tolist()

    if dataset == "tcv_11":
        ds_path = sample_ds_path
        cheat_training = True
    elif dataset == "tcv_132":
        ds_path = os.path.join(
            get_path_to_ml_data_dump(), "popsim_studies", "transport_sample", "sample_small", "dataset_full", "sample_small.zarr"
        )
        cheat_training = False
    else:
        raise ValueError(f"Invalid dataset option {dataset}. Must be one of 'tcv_11', 'tcv_132'.")

    # Point the base config to the sample transport dataset
    transport_predictor_config = BASE_CONFIG.model_copy(
        update={
            "debug": False,
            "max_epochs": MAX_EPOCHS,
            "epochs_per_val": EPOCHS_PER_VAL,
            "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "transport_predictor"),
            "dataloader_config": {
                **BASE_CONFIG.dataloader_config,
                "ds_path": ds_path,
                "test_shots": test_shots,
                "cheat_training": cheat_training,
            },
            "model_init_config": {
                **BASE_CONFIG.model_init_config,
                "restore_submodules": True,
            },
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
        ],
    )

    return transport_predictor_config


########################################
# Section 2.2: Train profile predictor #
########################################
def train_profile_predictor(clean: bool = False, dataset: str = "tcv_11"):
    transport_predictor_config = setup_main_config(dataset=dataset)
    profile_predictor_config = TrainConfig.load(transport_predictor_config.model_init_config["submodules"]["profile_predictor"])
    profile_predictor_config = profile_predictor_config.model_copy(
        update={
            "max_epochs": MAX_EPOCHS,
            "epochs_per_val": EPOCHS_PER_VAL,
            "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "profile_predictor"),
        }
    )
    if not os.path.exists(profile_predictor_config.checkpoint_dir) or clean:
        shutil.rmtree(profile_predictor_config.checkpoint_dir, ignore_errors=True)
        launch_train(profile_predictor_config.model_dump(), use_wandb=False)
    else:
        logger.info(f"Profile predictor checkpoint directory {profile_predictor_config.checkpoint_dir} already exists. Skipping training.")


####################################
# Section 2.3: Train power balance #
####################################
def train_power_balance(clean: bool = False, dataset: str = "tcv_11"):
    transport_predictor_config = setup_main_config(dataset=dataset)
    power_balance_config = TrainConfig.load(transport_predictor_config.model_init_config["submodules"]["power_balance"])
    power_balance_config = power_balance_config.model_copy(
        update={
            "max_epochs": MAX_EPOCHS,
            "epochs_per_val": EPOCHS_PER_VAL,
            "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "power_balance"),
        }
    )
    if not os.path.exists(power_balance_config.checkpoint_dir) or clean:
        shutil.rmtree(power_balance_config.checkpoint_dir, ignore_errors=True)
        launch_train(power_balance_config.model_dump(), use_wandb=False)
    else:
        logger.info(f"Power balance checkpoint directory {power_balance_config.checkpoint_dir} already exists. Skipping training.")


#############################################
# Section 2.4: OhmicPower and RadiatedPower #
#############################################
def train_ohmic_power(clean: bool = False, dataset: str = "tcv_11"):
    transport_predictor_config = setup_main_config(dataset=dataset)
    ohmic_power_config = TrainConfig.load(transport_predictor_config.model_init_config["submodules"]["p_oh_predictor"])
    ohmic_power_config = ohmic_power_config.model_copy(
        update={
            "max_epochs": MAX_EPOCHS,
            "epochs_per_val": EPOCHS_PER_VAL,
            "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "p_oh_predictor"),
        }
    )
    if not os.path.exists(ohmic_power_config.checkpoint_dir) or clean:
        shutil.rmtree(ohmic_power_config.checkpoint_dir, ignore_errors=True)
        launch_train(ohmic_power_config.model_dump(), use_wandb=False)
    else:
        logger.info(f"Ohmic power checkpoint directory {ohmic_power_config.checkpoint_dir} already exists. Skipping training.")


def train_radiated_power(clean: bool = False, dataset: str = "tcv_11"):
    transport_predictor_config = setup_main_config(dataset=dataset)
    radiated_power_config = TrainConfig.load(transport_predictor_config.model_init_config["submodules"]["p_rad_predictor"])
    radiated_power_config = radiated_power_config.model_copy(
        update={
            "max_epochs": MAX_EPOCHS,
            "epochs_per_val": EPOCHS_PER_VAL,
            "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "p_rad_predictor"),
        }
    )
    if not os.path.exists(radiated_power_config.checkpoint_dir) or clean:
        shutil.rmtree(radiated_power_config.checkpoint_dir, ignore_errors=True)
        launch_train(radiated_power_config.model_dump(), use_wandb=False)
    else:
        logger.info(f"Radiated power checkpoint directory {radiated_power_config.checkpoint_dir} already exists. Skipping training.")


############################
# 2.5 Main module training #
############################
def train_transport_predictor(clean: bool = False, dataset: str = "tcv_11"):
    transport_predictor_config = setup_main_config(dataset=dataset)

    transport_predictor_config = transport_predictor_config.model_copy(
        update={
            "model_init_config": {
                **transport_predictor_config.model_init_config,
                "submodules": {
                    "profile_predictor": {
                        **transport_predictor_config.model_init_config["submodules"]["profile_predictor"],
                        "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "profile_predictor"),
                    },
                    "power_balance": {
                        **transport_predictor_config.model_init_config["submodules"]["power_balance"],
                        "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "power_balance"),
                    },
                    "p_oh_predictor": {
                        **transport_predictor_config.model_init_config["submodules"]["p_oh_predictor"],
                        "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "p_oh_predictor"),
                    },
                    "p_rad_predictor": {
                        **transport_predictor_config.model_init_config["submodules"]["p_rad_predictor"],
                        "checkpoint_dir": os.path.join(CHECKPOINT_DIR_BASE, dataset, "p_rad_predictor"),
                    },
                },
            }
        }
    )

    if not os.path.exists(transport_predictor_config.checkpoint_dir) or clean:
        shutil.rmtree(transport_predictor_config.checkpoint_dir, ignore_errors=True)
        launch_train(transport_predictor_config.model_dump(), use_wandb=False)
    else:
        logger.info(
            f"Transport predictor checkpoint directory {transport_predictor_config.checkpoint_dir} already exists. Skipping training."
        )


def save_checkpoint(dataset: str = "tcv_11"):
    transport_predictor_config = setup_main_config(dataset=dataset)
    checkpoint_dir = transport_predictor_config.checkpoint_dir
    output_path = os.path.join(CHECKPOINT_DIR_BASE, f"sample_checkpoint_{dataset}")
    shutil.make_archive(output_path, "zip", checkpoint_dir)


def eval_transport_predictor(dataset: str = "tcv_11"):
    logger.info("Running evaluation for transport predictor demo...")
    transport_predictor_config = setup_main_config(dataset=dataset)
    transport_predictor_config = transport_predictor_config.model_copy(
        update={
            "model_init_config": {
                **transport_predictor_config.model_init_config,
                "restore_submodules": False,  # Don't need to restore submodules because we're restoring the whole model
            }
        }
    )
    _, train_dl, _, test_dl = TransportPredictorTrainRunBuilder.get_dataloaders(transport_predictor_config.dataloader_config)
    trainer = Trainer(
        model=TransportPredictorTrainRunBuilder.model_init(train_dl, transport_predictor_config.model_init_config),
        loss_fn=TransportPredictorTrainRunBuilder.get_loss_fn(transport_predictor_config.loss_config),
        optimizer=TransportPredictorTrainRunBuilder.get_optimizer(transport_predictor_config.optimizer_config),
        checkpoint_dir=transport_predictor_config.checkpoint_dir,
    )
    trainer.restore_best_checkpoint()
    logger.info("Restored best checkpoint for evaluation.")
    eval_data = trainer.run_evals(test_dl)

    input_ds = eval_data.input_ds.reset_index("sample")
    output_ds = eval_data.output_ds.reset_index("sample")

    input_ds.to_netcdf(os.path.join(CHECKPOINT_DIR_BASE, dataset, "eval_input_data.nc"))
    output_ds.to_netcdf(os.path.join(CHECKPOINT_DIR_BASE, dataset, "eval_output_data.nc"))
    logger.debug(f"Saved evaluation data to {CHECKPOINT_DIR_BASE}")


def train_all(clean: bool = False, dataset: str = "tcv_11"):
    logger.info("Running training for transport predictor demo...")
    train_profile_predictor(clean=clean, dataset=dataset)
    train_ohmic_power(clean=clean, dataset=dataset)
    train_radiated_power(clean=clean, dataset=dataset)
    train_power_balance(clean=clean, dataset=dataset)
    train_transport_predictor(clean=clean, dataset=dataset)
    save_checkpoint(dataset=dataset)
    logger.info("Training for transport predictor demo completed.")


if __name__ == "__main__":
    # Usage: python transport_sample_checkpoint.py <command> [--clean] [--dataset <dataset_name>]
    fire.Fire(
        {
            "train_profile_predictor": train_profile_predictor,
            "train_ohmic_power": train_ohmic_power,
            "train_radiated_power": train_radiated_power,
            "train_power_balance": train_power_balance,
            "train_transport_predictor": train_transport_predictor,
            "train_all": train_all,
            "eval_transport_predictor": eval_transport_predictor,
            "save_checkpoint": save_checkpoint,
        }
    )
