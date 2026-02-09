from popsim.data import get_path_to_ml_data_dump
from popsim.ml.rtd_activation import Activation
from popsim.modules.meqml.fbt.trb import FBTTrainRunBuilder

FBT_SURROGATE_CONFIG = {
    "project": "fbt_surrogate_tcv",
    "train_run_builder": FBTTrainRunBuilder,
    "max_epochs": 5000,
    "epochs_per_val": 10,
    "checkpoint_dir": None,
    "dataloader_config": {
        "path": get_path_to_ml_data_dump() / "TCV/caches/fbt_jan2026.zarr",
        "convert_xr_to_jnp": False,
        "time_coord": "time",
        "episode_coord": "shot",
        "split_fracs": (0.7, 0.15, 0.15),
        "batch_size": 16384,
        "key": 42,
        "input_vars": [
            "LY.SC.lcC",
            "LY.SC.lcL",
            "LY.SC.lcS",
            "LY.SC.lcX1",
            "LY.SC.lcX2",
            "LY.SC.lcD",
            "LY.SC.lcI",
            "LY.SC.rc",
            "LY.SC.zc",
            "LY.Ip",
            "LY.bp",
            "LY.qA",
            "LY.rBt",
            "L.G.rl",
            "L.G.zl",
        ],
        "target_vars": [
            "LY.Ia",
            "LY.MpOH",
            "LY.MpEF",
            "loss_weight",
        ],
        "extra_vars": [],
    },
    "model_init_config": {
        "nn_depth": 3,
        "nn_width": 128,
        "n_latent": 13,
        "n_heads": 5,
        "layernorm": True,
        "attention_input_scale": 1.0,
        "dropout_rate": 0.33,
        "prng_seed": 42,
        "activation": Activation.RELU.value,
    },
    "loss_config": {
        "huber_delta": 0.1,
        "coil_current_norm": 7500.0,  # Normalizing factor for coil current in the loss function.
    },
    "optimizer_config": {
        "lr0": 5e-4,
        "lr_peak": 1e-2,
        "warmup_steps": 100,
        "decay_steps": 20000,
        "lrf": 5e-3,
        "weight_decay": 5e-3,
    },
}

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/loss.mean", "goal": "minimize"},
    "parameters": {
        "model_init_config": {
            "parameters": {
                "nn_width": {"values": [128, 256]},
                "nn_depth": {"values": [2, 3, 4]},
                "n_latent": {"min": 10, "max": 18},
                "n_heads": {"min": 4, "max": 8},
                "attention_input_scale": {"min": 0.5, "max": 2.0},
                "dropout_rate": {"min": 0.0, "max": 0.5},
            },
        }
    },
}
