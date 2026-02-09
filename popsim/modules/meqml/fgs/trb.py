import jax.numpy as jnp
import optax
import xarray as xr

from popsim.ml import DataLoader, TrainRunBuilder, make_standard_dataloaders
from popsim.modules.meqml.fgs import evals
from popsim.modules.meqml.fgs.module import FGS


class FGSTrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(config: dict) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """
        ds = xr.open_dataset(config["path"]).load()

        ds = ds.set_coords("time")

        # Remove "path" from config.
        del config["path"]
        train_dl, val_dl, test_dl = make_standard_dataloaders(ds=ds, **config)
        return ds, train_dl, val_dl, test_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> FGS:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        module = FGS.init(train_dl, **model_init_config)
        return module

    @staticmethod
    def get_loss_fn(config: dict):
        def loss_fn(pred, targ):
            def fn(p, t):
                error_norm = (p - t) / jnp.linalg.norm(t)
                return jnp.sum(optax.huber_loss(error_norm, delta=config["huber_delta"]))

            Fx_loss = fn(pred.Fx.data, targ["LY.Fx"].data)
            Ff_loss = fn(pred.Ff.data, targ["LY.Ff"].data)
            Bm_loss = fn(pred.Bm.data, targ["LY.Bm"].data)
            return Fx_loss + Ff_loss + Bm_loss

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
            "median_relative_error": evals.relative_error_medians,
        }
        return suite

    @staticmethod
    def get_test_eval_suite(config: dict):
        suite = {
            "median_relative_error": evals.relative_error_medians,
            "padded_relative_error_hists": evals.padded_relative_error_hists,
        }
        return suite
