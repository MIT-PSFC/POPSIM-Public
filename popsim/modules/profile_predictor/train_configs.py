from popsim.modules.profile_predictor.module import ShapeType
from popsim.modules.profile_predictor.training_run_builder import ProfilePredictorTrainRunBuilder

DUMMY_CONFIG = {
    "project": "dummy_profile_predictor",
    "train_run_builder": ProfilePredictorTrainRunBuilder,
    "max_epochs": 500,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "dataloader_config": {
        "ds": "dummy",
        "debug": False,
        "input_vars": ["Ip_MA", "a_minor", "kappa", "delta", "Paux_MW", "ne20_line_avg", "Wtot_MJ", "B0", "R0", "ne20_edge"],
        "target_vars": ["ne20_rho", "Te_keV_rho"],
        "extra_vars": ["Te_shape", "ne_shape"],
        "split_fracs": (0.5, 0.3, 0.2),
        "prng_seed": 42,
        "batch_size": 4096,
        "convert_xr_to_jnp": False,
    },
    "model_init_config": {
        "shape_type": ShapeType.CONVEX_COMBINATION.value,
        "te_shape_var": "Te_shape",
        "ne_shape_var": "ne_shape",
        "n_shapes": 3,
        "nn_depth": 2,
        "nn_width": 16,
        "softmax_temp": 1,
        "use_ne_edge": False,
        "freeze_shapes": True,
        "prng_seed": 42,
    },
    "loss_config": {
        "huber_delta": 0.5,
    },
    "optimizer_config": {
        "lr0": 3e-3,
        "transition_steps": 500,
        "decay_rate": 0.5,
        "lrf": 5e-4,
        "weight_decay": 2e-4,
    },
    "trainable_getter_config": {
        "freeze_shapes": True,
    },
    "test_eval_suite_config": {
        "device": "dummy",
    },
}


# TCV_CONFIG is a copy of DUMMY_CONFIG with the "project" and "ds" fields changed
TCV_CONFIG = DUMMY_CONFIG.copy()
TCV_CONFIG["project"] = "tcv_profile_predictor"
TCV_CONFIG["ds"] = "tcv"
TCV_CONFIG["test_eval_suite_config"] = {
    "device": "tcv",
}
