from popsim.data import get_path_to_ml_data_dump
from popsim.ml.rtd_activation import Activation
from popsim.modules.meqml.fast_obs.trb import FastObsTrainRunBuilder

FAST_OBS_CONFIG = {
    "project": "fast_obs_tcv",
    "train_run_builder": FastObsTrainRunBuilder,
    "max_epochs": 1000,
    "epochs_per_val": 10,
    "checkpoint_dir": None,
    "dataloader_config": {
        "path": get_path_to_ml_data_dump() / "TCV/caches/liuqe.zarr",
        "convert_xr_to_jnp": False,
        "time_coord": "time",
        "episode_coord": "shot",
        "split_fracs": (0.7, 0.15, 0.15),
        "batch_size": 2**15,
        "key": 42,
        "input_vars": [
            "LX.Uf",
            "LX.Ff",
            "LX.Bm",
            "LX.Ia",
            "LX.Ft",
            "LX.rBt",
        ],
        "target_vars": ["LY.rIp", "LY.Ip", "LY.zIp"],
        "extra_vars": [],
    },
    "model_init_config": {
        "nn_depth": 2,
        "nn_width": 16,
        "n_latent": 23,
        "dropout_rate": 0.5,
        "activation": Activation.RELU.value,
        "prng_seed": 42,
    },
    "loss_config": {
        "huber_delta": 0.05,
    },
    "optimizer_config": {
        "lr0": 5e-3,
        "lr_peak": 2e-2,
        "warmup_steps": 100,
        "decay_steps": 1000,
        "lrf": 3e-3,
        "weight_decay": 5e-4,
    },
}


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/loss.mean", "goal": "minimize"},
    "parameters": {
        "model_init_config": {
            "parameters": {
                "nn_width": {"values": [16, 32, 64]},
                "nn_depth": {"values": [1, 2]},
                "n_latent": {"min": 3, "max": 32},
                "dropout_rate": {"min": 0.0, "max": 0.5},
            },
        }
    },
}
