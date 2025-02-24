import os

from popsim.modules.profile_predictor.evals import compute_integrated_error
from popsim.modules.profile_predictor.module import ShapeType

SPARC_CONFIG = {
    "project": "sparc_profile_predictor",
    "ds": "sparc",
    "debug": False,
    "use_wandb": False,
    "lr0": 3e-3,
    "transition_steps": 500,
    "decay_rate": 0.5,
    "lrf": 5e-4,
    "weight_decay": 2e-4,
    "max_epochs": 500,
    "epochs_per_val": 10,
    "nn_depth": 1,
    "nn_width": 16,
    "prng_seed": 42,
    "n_shapes": 3,
    "n_basis": 10,
    "batch_size": 4096,
    "split_fracs": (0.5, 0.3, 0.2),
    "huber_delta": 0.5,
    "shape_type": ShapeType.CONVEX_COMBINATION.value,
    "use_ne_edge": False,
    "freeze_shapes": True,
    "softmax_temp": 10,
    "input_vars": ["Ip_MA", "a_minor", "kappa", "delta", "Paux_MW", "ne20_line_avg", "Wtot_MJ", "B0", "R0", "ne20_edge"],
    "target_vars": ["ne20_rho", "Te_keV_rho"],
    "extra_vars": ["Te_shape", "ne_shape"],
    "train_eval_suite": {"integrated_profile_error": compute_integrated_error},
    "checkpoint_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "sparc_latest"),
}

# TCV_CONFIG is a copy of SPARC_CONFIG with the "project" and "ds" fields changed
TCV_CONFIG = SPARC_CONFIG.copy()
TCV_CONFIG["project"] = "tcv_profile_predictor"
TCV_CONFIG["ds"] = "tcv"
