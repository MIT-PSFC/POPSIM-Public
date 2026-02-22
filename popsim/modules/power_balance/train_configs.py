from popsim.ml.train_config import TrainConfig
from popsim.simulate import StepperType

BASE_CONFIG = TrainConfig(
    project="power_balance_base",
    train_run_builder="popsim.modules.power_balance.training_run_builder.PowerBalanceTrainRunBuilder",
    max_epochs=2,
    epochs_per_val=1,
    checkpoint_dir=None,
    dataloader_config={
        "ds": None,
        "debug": True,
        "state_vars": ["Wtot_MJ"],
        "input_vars": ["Ip_MA", "B0", "ne19", "P_abs_MW", "R0", "kappa", "epsilon", "delta_top", "delta_bottom", "P_rad_MW"],
        "target_vars": ["Wtot_MJ"],
    },
    model_init_config={
        "stepper": StepperType.SIMPLE_EULER,
        "model_type": None,
        "min_val": 0.001,  # minimum reasonable value for tau_e to avoid division by zero
        "max_val": 0.12,  # seconds
    },
    loss_config={
        "huber_delta": 0.5,
    },
    optimizer_config={
        "lr0": 3e-3,
        "transition_steps": 500,
        "decay_rate": 0.5,
        "lrf": 5e-4,
        "weight_decay": 2e-4,
    },
)

NEURAL_NETWORK_CONFIG = BASE_CONFIG.model_copy(
    update={
        "project": "power_balance_nn",
        "model_init_config": {
            **BASE_CONFIG.model_init_config,
            "model_type": "neural_network",
            "network_vars": ["Ip_MA", "B0", "ne19", "P_abs_MW", "R0", "kappa", "epsilon", "delta_top", "delta_bottom"],
            "nn_depth": 2,
            "nn_width": 16,
            "prng_seed": 42,
        },
    }
)

SCALING_LAW_CONFIG = BASE_CONFIG.model_copy(
    update={
        "project": "power_balance_scaling_law",
        "model_init_config": {
            **BASE_CONFIG.model_init_config,
            "model_type": "scaling_law",
        },
    }
)
