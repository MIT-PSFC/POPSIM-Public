import jax.numpy as jnp
import optax
import xarray as xr

from popsim.ml import DataLoader, TrainRunBuilder, make_standard_dataloaders
from popsim.modules.meqml.fbt import evals
from popsim.modules.meqml.fbt.module import FBTSurrogate


class FBTTrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(config: dict) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """
        ds = xr.open_dataset(config["path"]).set_coords("time").load()

        ds["loss_weight"] = xr.ones_like(ds["LY.bp"])

        # Remove "path" from config.
        del config["path"]
        train_dl, val_dl, test_dl = make_standard_dataloaders(ds=ds, **config)
        return ds, train_dl, val_dl, test_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> FBTSurrogate:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        module = FBTSurrogate.init(train_dl, **model_init_config)
        return module

    @staticmethod
    def get_loss_fn(config: dict):
        def loss_fn(pred, targ):
            Ia_pred = pred.Ia.data
            Ia_targ = targ["LY.Ia"].data

            Ia_error_norm = jnp.abs(Ia_pred - Ia_targ) / config["coil_current_norm"]

            prediction_loss = jnp.mean(optax.huber_loss(Ia_error_norm, delta=config["huber_delta"]))

            return prediction_loss

        return loss_fn

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=config["lr0"],
            peak_value=config["lr_peak"],
            warmup_steps=config["warmup_steps"],
            decay_steps=config["decay_steps"],
            end_value=config["lrf"],
        )
        opt = optax.adamw(learning_rate=schedule, weight_decay=config["weight_decay"])
        return opt

    @staticmethod
    def get_trainable_getter(config: dict):
        """Optionally return a function that takes in the trainable parameters of your model and returns the trainable parameters."""
        return None

    @staticmethod
    def get_val_eval_suite(config: dict):
        suite = {
            "mean_coil_errors": evals.eval_ia_prediction_error_quantiles,
        }
        return suite

    @staticmethod
    def get_test_eval_suite(config: dict):
        suite = {
            "eval_ia_prediction_errors": evals.eval_ia_prediction_errors,
            "eval_ia_pred_targ": evals.eval_ia_pred_targ,
            "eval_sorted_gui": evals.eval_sorted_gui,
            "mean_error_hist": evals.mean_error_hist,
        }
        return suite
