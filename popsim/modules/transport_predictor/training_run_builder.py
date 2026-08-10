from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import optax
import xarray as xr

from popsim.ml import DataLoader, IntegralLoss, TrainRunBuilder
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.split_utils import split_dataset_by_fracs, split_dataset_by_vals
from popsim.modules.transport_predictor.data import get_ds
from popsim.modules.transport_predictor.module import TransportPredictor, TransportPredictorEnv


class TransportPredictorTrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(dataloader_config: dict) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """

        ds, episode_coord = get_ds(dataloader_config["ds_path"], debug=dataloader_config["debug"])

        if dataloader_config.get("cheat_training", False):
            ds_train = ds
            ds_val = ds
            ds_test = ds
        else:
            # If not cheating, make sure to exclude test shots from training and validation sets
            test_shots = dataloader_config.get("test_shots")
            split_fracs = dataloader_config["split_fracs"]
            if test_shots:
                split_result, ds_reduced = split_dataset_by_vals(ds, vals=[test_shots], dim=episode_coord)
                ds_test = split_result[0]  # Assuming split_result is a list of datasets
                if len(split_fracs) == 2:
                    # Assuming that the provided split_fracs are for train and val, so we're fine
                    split_fracs_train_val = split_fracs
                elif len(split_fracs) == 3:
                    # Assuming the provided split_fracs are for train, val, and test, so need to renormalize
                    split_fracs_train_val = (split_fracs[0] / sum(split_fracs[:2]), split_fracs[1] / sum(split_fracs[:2]))
                else:
                    raise ValueError(f"Invalid number of split_fracs provided: {len(split_fracs)}. Must be 2 or 3.")
                ds_train, ds_val = split_dataset_by_fracs(
                    ds_reduced, fracs=split_fracs_train_val, dim=episode_coord, seed=dataloader_config["prng_seed"]
                )
                # Double check no test shots are in training or validation sets
                for shot in test_shots:
                    if shot in ds_train[episode_coord]:
                        raise ValueError(f"Test shot {shot} found in training set after splitting.")
                    if shot in ds_val[episode_coord]:
                        raise ValueError(f"Test shot {shot} found in validation set after splitting.")
            else:
                if len(split_fracs) != 3:
                    raise ValueError(
                        f"split_fracs must have 3 values if no test_shots are specified, but {len(split_fracs)} were provided."
                    )
                ds_train, ds_val, ds_test = split_dataset_by_fracs(
                    ds, fracs=split_fracs, dim=episode_coord, seed=dataloader_config["prng_seed"]
                )

        train_dl, val_dl, test_dl = make_dataloaders(
            datasets=[ds_train, ds_val, ds_test],
            time_coord="time",
            episode_coord=episode_coord,
            input_vars=dataloader_config["input_vars"],
            target_vars=dataloader_config["target_vars"],
            extra_vars=dataloader_config.get("extra_vars"),
            state_init_vars=dataloader_config.get("state_vars"),
            convert_xr_to_jnp=dataloader_config["convert_xr_to_jnp"],
            nan_handling="drop_slice_any",
        )

        return ds, train_dl, val_dl, test_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        rhogrid = jnp.asarray(train_dl.ds["rho"].data)  # , dtype=DATA_TYPE)
        submodule_configs = model_init_config["submodules"]
        module = TransportPredictor.init(
            profile_predictor_config=submodule_configs["profile_predictor"],
            power_balance_config=submodule_configs["power_balance"],
            p_oh_predictor_config=submodule_configs["p_oh_predictor"],
            p_rad_predictor_config=submodule_configs["p_rad_predictor"],
            rhogrid=rhogrid,
            restore_submodules=model_init_config.get("restore_submodules", False),
        )

        # Wrap the module in an environment since it's time-dependent
        env = TransportPredictorEnv(module=module)

        if model_init_config.get("restore_main_module", False):
            manager = create_default_checkpoint_manager(model_init_config["checkpoint_dir"])
            env = restore_model(manager, env)

        return env

    @staticmethod
    def get_loss_fn(loss_config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            # Profile predictor losses
            # ne, te, and rho are xr.Variables, so unwrap the underlying arrays for optax.
            ne_rho_loss = (
                jnp.trapezoid(
                    optax.huber_loss(pred.profile_predictor_output.ne.data, targ["ne20_rho"].data, delta=loss_config["huber_delta"]),
                    x=pred.rho.data,
                )
                * targ["fresh_profiles"].data
            )  # Only apply loss to time steps with fresh profile measurements
            te_rho_loss = (
                jnp.trapezoid(
                    optax.huber_loss(pred.profile_predictor_output.te.data, targ["Te_keV_rho"].data, delta=loss_config["huber_delta"]),
                    x=pred.rho.data,
                )
                * targ["fresh_profiles"].data
            )  # Only apply loss to time steps with fresh profile measurements

            # Other losses
            stored_energy_error = jnp.abs(pred.power_balance_output.Wtot_MJ_pred - targ["Wtot_MJ"].data)
            stored_energy_loss = optax.huber_loss(stored_energy_error, delta=loss_config["huber_delta"])
            # Ohmic power losses
            ohmic_error = jnp.abs(pred.p_oh_output.P_oh_MW_pred - targ["P_oh_MW"].data)
            p_oh_loss = optax.huber_loss(ohmic_error, delta=loss_config["huber_delta"])
            # Radiated power losses
            radiated_error = jnp.abs(pred.p_rad_output.P_rad_MW_pred - targ["P_rad_MW"].data)
            p_rad_loss = optax.huber_loss(radiated_error, delta=loss_config["huber_delta"])

            # Combine losses according to loss_config weights
            loss_weight = loss_config["loss_weight"]
            total_loss = (
                loss_weight["ne20_rho"] * ne_rho_loss
                + loss_weight["Te_keV_rho"] * te_rho_loss
                + loss_weight["Wtot_MJ"] * stored_energy_loss
                + loss_weight["P_oh_MW"] * p_oh_loss
                + loss_weight["P_rad_MW"] * p_rad_loss
            )

            return total_loss

        return IntegralLoss(loss_fn)

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        schedule = optax.exponential_decay(
            init_value=config["lr0"],
            transition_steps=config["transition_steps"],
            decay_rate=config["decay_rate"],
            end_value=config["lrf"],
        )
        opt = optax.adamw(learning_rate=schedule, weight_decay=config["weight_decay"])
        return opt
