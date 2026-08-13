import os

import pytest

from popsim import DATA_DIR
from popsim.ml.launch import launch_train
from popsim.modules.transport_predictor.train_configs import BASE_CONFIG


def test_oneshot_training():
    # Test training for the primary module and all submodules at once
    TEST_CONFIG = BASE_CONFIG.model_copy(
        update={
            "name": f"test_transport_predictor",
            "max_epochs": 2,
            "epochs_per_val": 1,
            "dataloader_config": {
                **BASE_CONFIG.dataloader_config,
                "ds_path": os.path.join(DATA_DIR, "tcv", "scrambled_transport_sample.nc"),
                "debug": True,
                "segment_length_train": 20,
                "segment_overlap_train": 0,
                "batch_size": 256,
                "cheat_training": True,  # Transport sample is too small to split, just use same dataset for all dataloaders
            }
        },
    )

    trainer, _, _, test_dl, _ = launch_train(TEST_CONFIG.model_dump(), use_wandb=False)
    assert trainer is not None
    assert test_dl is not None
