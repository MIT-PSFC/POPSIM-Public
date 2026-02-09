from popsim.data import get_path_to_ml_data_dump
from popsim.modules.meqml.liuqe.trb import LIUQETrainRunBuilder

LIUQE_CONFIG = {
    "project": "liuqe_tcv",
    "train_run_builder": LIUQETrainRunBuilder,
    "max_epochs": 50,
    "epochs_per_val": 10,
    "checkpoint_dir": None,  # To be overwritten at wandb run initialization.
    "dataloader_config": {
        "path": get_path_to_ml_data_dump() / "TCV/caches/liuqe_small.zarr",
        "convert_xr_to_jnp": False,
        "time_coord": "time",
        "episode_coord": "shot",
        "split_fracs": (0.7, 0.15, 0.15),
        "batch_size": 4096,
        "key": 42,
        "input_vars": [
            "LX.Uf",
            "LX.Ff",
            "LX.Bm",
            "LX.Ia",
            "LX.Ft",
            "LX.rBt",
        ],
        "target_vars": ["LY.Fx", "LY.Ip", "LY.bp", "LY.iqQ", "LY.qA"],
        "extra_vars": [
            "LY.rq",
            "LY.zq",
            "LY.kappa",
            "LY.deltau",
            "LY.deltal",
            "LY.delta",
            "LY.aminor",
            "LY.rgeom",
            "LY.zgeom",
            "L.G.rl",
            "L.G.zl",
        ],
    },
    "model_init_config": {
        "n_latent": 50,
        "use_nn": False,
        "nn_depth": 2,
        "nn_width": 128,
        "prng_seed": 42,
    },
    "loss_config": {
        "Fx_weight": 10.0,
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
