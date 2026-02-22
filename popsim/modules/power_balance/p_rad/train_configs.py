from popsim.ml.train_config import TrainConfig

BASE_CONFIG = TrainConfig(
    project="p_rad_base",
    train_run_builder="popsim.modules.power_balance.p_rad.training_run_builder.RadiatedPowerTrainRunBuilder",
    max_epochs=2,
    epochs_per_val=1,
    checkpoint_dir=None,
    dataloader_config={
        "ds": None,
        "debug": True,
        "input_vars": ["Ip_MA", "R0", "a_minor", "kappa", "delta_top", "delta_bottom", "ne20", "Wtot_MJ"],
        "target_vars": ["P_rad_MW"],
    },
    model_init_config={
        "nn_depth": 2,
        "nn_width": 16,
        "min_val": 0,  # Min radiated power in MW
        "max_val": 2,  # Max radiated power in MW
        "prng_seed": 42,
        "in_size": 8,
        "out_size": 1,
    },
    loss_config={
        "huber_delta": 0.5,
    },
    optimizer_config={
        "lr0": 5e-3,
        "transition_steps": 1000,
        "decay_rate": 0.5,
        "lrf": 2e-3,
        "weight_decay": 2e-4,
    },
)
