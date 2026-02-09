from popsim.data import get_path_to_ml_data_dump
from popsim.modules.meqml.fgs.trb import FGSTrainRunBuilder

FGS_CONFIG = {
    "project": "fgs_tcv",
    "train_run_builder": FGSTrainRunBuilder,
    "max_epochs": 1000,
    "epochs_per_val": 10,
    "checkpoint_dir": None,
    "dataloader_config": {
        "path": get_path_to_ml_data_dump() / "TCV/caches/liuqe.zarr",
        "convert_xr_to_jnp": False,
        "time_coord": "time",
        "episode_coord": "shot",
        "split_fracs": (0.7, 0.15, 0.15),
        "batch_size": 2**16,
        "key": 42,
        "input_vars": ["LY.Ia", "LY.Iu", "LY.rBt", "LY.Ip", "LY.bp", "LY.qA"],
        "target_vars": ["LY.Fx", "LY.Ff", "LY.Bm"],
        "extra_vars": [],
    },
    "model_init_config": {
        "n_latent": 20,
        "nn_depth": 2,
        "nn_width": 512,
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
                "n_latent": {"values": [20, 30, 40, 50]},
                "nn_width": {"values": [128, 256, 512]},
                "nn_depth": {"values": [1, 2, 3]},
            },
        },
    },
}
