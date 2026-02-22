from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xarray as xr

from popsim.ml import DataLoader, TrainRunBuilder, make_standard_dataloaders
from popsim.modules.profile_predictor import evals
from popsim.modules.profile_predictor.data import get_ds
from popsim.modules.profile_predictor.module import ProfilePredictor, ShapeType, kmeans_initial_guess, pca_initial_guess


class ProfilePredictorTrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(config: dict) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """
        ds, episode_coord = get_ds(config["ds"], config["debug"])

        train_dl, val_dl, test_dl = make_standard_dataloaders(
            ds=ds,
            time_coord="time",
            episode_coord=episode_coord,
            input_vars=config["input_vars"],
            target_vars=config["target_vars"],
            extra_vars=config["extra_vars"],
            split_fracs=config["split_fracs"],
            key=config["prng_seed"],
            batch_size=config["batch_size"],
            convert_xr_to_jnp=config["convert_xr_to_jnp"],
        )
        return ds, train_dl, val_dl, test_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> Any:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """
        shape_type = model_init_config["shape_type"]
        te_shape_var = model_init_config["te_shape_var"]
        ne_shape_var = model_init_config["ne_shape_var"]
        n_shapes = model_init_config["n_shapes"]

        module = ProfilePredictor.init(
            n_shapes=model_init_config["n_shapes"],
            rhogrid=train_dl.ds["rho"].data,
            nn_width=model_init_config["nn_width"],
            nn_depth=model_init_config["nn_depth"],
            shape_type=shape_type,
            softmax_temp=model_init_config["softmax_temp"],
            use_ne_edge=model_init_config["use_ne_edge"],
            prng_seed=model_init_config["prng_seed"],
        )

        # PCA/K-means initial guess for the shapes.
        ds = train_dl.ds
        sample_dim = train_dl.dataset.training_metadata.sample_dim

        if shape_type == ShapeType.PCA_LIKE:
            te_shapes, ne_shapes = pca_initial_guess(n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim)
        elif shape_type == ShapeType.CONVEX_COMBINATION:
            te_shapes, ne_shapes = kmeans_initial_guess(n_shapes, ds[te_shape_var], ds[ne_shape_var], sample_dim)
        else:
            raise ValueError(f"Invalid shape type: {shape_type}")

        # Overwrite the initial shapes in the module with the initial guess.
        module = eqx.tree_at(
            lambda m: (m.te_shapes, m.ne_shapes),
            module,
            (te_shapes, ne_shapes),
        )

        return module

    @staticmethod
    def get_loss_fn(config: dict) -> Callable[[Any, Any], jnp.ndarray]:
        def loss_fn(pred, targ):
            ne_rho_loss = jnp.trapezoid(
                optax.huber_loss(pred.ne.data, targ["ne20_rho"].data, delta=config["huber_delta"]), x=pred.ne.rho.data
            )
            te_rho_loss = jnp.trapezoid(
                optax.huber_loss(pred.te.data, targ["Te_keV_rho"].data, delta=config["huber_delta"]), x=pred.te.rho.data
            )
            return ne_rho_loss + te_rho_loss

        return loss_fn

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

    @staticmethod
    def get_trainable_getter(config: dict) -> Callable[[Any], Any] | None:
        """Optionally return a function that takes in the trainable parameters of your model and returns the trainable parameters."""
        freeze_shapes = config["freeze_shapes"]

        def get_trainable(module: ProfilePredictor):
            if freeze_shapes:
                # Get all leaves that are not a part of te_shapes and ne_shapes.
                # All of these leaves are trainable.
                ids_of_shape_leaves = [id(x) for x in jax.tree.leaves((module.te_shapes, module.ne_shapes))]
                return [x for x in jax.tree.leaves(module) if id(x) not in ids_of_shape_leaves]
            else:
                return module

        return get_trainable

    @staticmethod
    def get_val_eval_suite(config):
        return {"integrated_profile_error": evals.compute_integrated_error}

    @staticmethod
    def get_test_eval_suite(config):
        if config is None:  # No test eval suite config given, return None
            return None
        elif config["device"] == "sparc":
            test_eval_suite = {
                "violin_shapes_in_data": evals.violin_shapes_in_data,
                "integrated_profile_error": evals.compute_integrated_error,
                "compare_profiles": evals.compare_profiles_for_episode,
                "make_histograms": evals.make_histograms,
                "plot_shapes": evals.plot_shapes,
                "make_quantile_examples": evals.make_quantile_examples,
                "plot_ne_te_weights": evals.plot_ne_te_weights,
            }
        elif config["device"] == "tcv":
            test_eval_suite = {
                "violin_shapes_in_data": evals.violin_shapes_in_data,
            }
        else:
            raise ValueError(f"Invalid device: {config['device']}")
        return test_eval_suite
