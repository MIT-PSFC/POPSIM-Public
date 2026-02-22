from popsim.ml.train_config import TrainConfig
from popsim.modules.power_balance.p_oh.train_configs import BASE_CONFIG as OHMIC_POWER_CONFIG
from popsim.modules.power_balance.p_rad.train_configs import BASE_CONFIG as RADIATED_POWER_CONFIG
from popsim.modules.power_balance.train_configs import NEURAL_NETWORK_CONFIG as POWER_BALANCE_CONFIG
from popsim.modules.profile_predictor.train_configs import TCV_CONFIG as PROFILE_PREDICTOR_CONFIG
from popsim.modules.transport_predictor.training_run_builder import TransportPredictorTrainRunBuilder

########################################################################
# MODIFICATIONS FOR SUBMODULES BEFORE THEY'RE PUT INTO THE MAIN MODULE #
########################################################################

# TODO(ZanderKeith): Deal with ne20_edge properly
PROFILE_PREDICTOR_CONFIG = TrainConfig(**PROFILE_PREDICTOR_CONFIG)
PROFILE_PREDICTOR_CONFIG = PROFILE_PREDICTOR_CONFIG.model_copy(
    update={
        "dataloader_config": {
            **PROFILE_PREDICTOR_CONFIG.dataloader_config,
            "input_vars": ["Ip_MA", "a_minor", "kappa", "delta", "Paux_MW", "ne20_line_avg", "Wtot_MJ", "B0", "R0"],
        },
        "test_eval_suite_config": None,
    }
)

###############################
# CONFIGS FOR THE MAIN MODULE #
###############################

BASE_CONFIG = TrainConfig(
    project="transport_predictor_base",
    train_run_builder=TransportPredictorTrainRunBuilder,
    max_epochs=2,
    epochs_per_val=1,
    dataloader_config={
        "ds_path": None,
        "debug": False,
        "state_vars": ["Wtot_MJ"],
        "input_vars": ["R0", "B0", "Ip_MA", "a_minor", "kappa", "delta_top", "delta_bottom", "P_aux_MW", "ne20"],
        "target_vars": ["ne20_rho", "Te_keV_rho", "Wtot_MJ", "P_oh_MW", "P_rad_MW", "fresh_profiles"],
        "extra_vars": ["ne_shape", "Te_shape"],
        "convert_xr_to_jnp": False,
        "split_fracs": (0.64, 0.16, 0.2),
        "ext_method": "chronological",
        "prng_seed": 42,
        # Hyperparameters
        "segment_length_train": 50,
        "segment_overlap_train": 0,
        "batch_size": 8192,
        # Part of validation, should be left alone during hyperparameter tuning
        "segment_length_val": None,
        "segment_overlap_val": 0,
    },
    model_init_config={
        "submodules": {
            "profile_predictor": PROFILE_PREDICTOR_CONFIG,
            "power_balance": POWER_BALANCE_CONFIG,
            "p_oh_predictor": OHMIC_POWER_CONFIG,
            "p_rad_predictor": RADIATED_POWER_CONFIG,
        },
        "restore_submodules": False,
    },
    loss_config={
        "huber_delta": 0.5,
        "loss_weight": {
            "ne20_rho": 500,  # Profile measurement once every ~50 ms, so weight more heavily
            "Te_keV_rho": 500,  # Profile measurement once every ~50 ms, so weight more heavily
            "Wtot_MJ": 800,  # Wtot is ~0.01 MJ so we need a higher weight to make sure this doesn't drift too much
            "P_oh_MW": 0.5,
            "P_rad_MW": 0.5,
        },
    },
    optimizer_config={
        "lr0": 1e-4,
        "transition_steps": 500,
        "decay_rate": 0.5,
        "lrf": 5e-4,
        "weight_decay": 2e-4,
    },
)


def update_submodule_configs(main_config: dict, submodules: list[str]) -> TrainConfig:
    new_submodule_configs = {}
    for submodule in submodules:
        submodule_config = main_config["model_init_config"]["submodules"][submodule]
        if not isinstance(submodule_config, dict):
            submodule_config = submodule_config.model_dump()

        submodule_config["project"] = f"{main_config['project']}.{submodule_config['project']}"

        # Set the data_train_run_builder for the submodules to match the main module's train_run_builder.
        submodule_config["dataloader_config"]["data_train_run_builder"] = main_config["train_run_builder"]

        # Ensure there is a perfect match between the dataloader configs of the main module and the submodules,
        # excepting the state_vars, input_vars, target_vars, and extra_vars which are specific to each submodule.
        for key in main_config["dataloader_config"].keys():
            if key not in ["state_vars", "input_vars", "target_vars", "extra_vars"]:
                submodule_config["dataloader_config"][key] = main_config["dataloader_config"][key]

        # Replacing target nans is only relevant to the main module
        submodule_config["dataloader_config"]["replace_target_nans"] = False

        new_submodule_configs[submodule] = submodule_config

    main_config["model_init_config"]["submodules"] = new_submodule_configs

    return TrainConfig(**main_config)


BASE_CONFIG = update_submodule_configs(
    BASE_CONFIG.model_dump(),
    [
        "profile_predictor",
        "power_balance",
        "p_oh_predictor",
        "p_rad_predictor",
    ],
)
