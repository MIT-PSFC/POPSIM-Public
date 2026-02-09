import jax
import optax
import xarray as xr

from popsim.math_utils import padded_relative_error
from popsim.ml import DataLoader, TrainRunBuilder, make_standard_dataloaders
from popsim.modules.meqml.liuqe import evals
from popsim.modules.meqml.liuqe.module import LIUQE


class LIUQETrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(config: dict) -> tuple[xr.Dataset, tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Get the dataset and dataloaders for training.
        """
        ds = xr.open_dataset(config["path"]).load().set_coords("time")

        # Remove "path" from config.
        del config["path"]
        train_dl, val_dl, test_dl = make_standard_dataloaders(ds=ds, **config)
        return ds, train_dl, val_dl, test_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> LIUQE:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        module = LIUQE.init(train_dl, **model_init_config)
        return module

    @staticmethod
    def get_loss_fn(config: dict):
        def loss_fn(pred, targ):
            errors = {
                "Fx": padded_relative_error(pred.Fx.data, targ["LY.Fx"].data),
                "Ip": padded_relative_error(pred.Ip.data, targ["LY.Ip"].data),
                "betap": padded_relative_error(pred.betap.data, targ["LY.bp"].data),
            }

            huber_losses = jax.tree.map(lambda x: optax.huber_loss(x, delta=config["huber_delta"]), errors)

            huber_losses["Fx"] *= config["Fx_weight"]

            loss = sum(huber_losses.values())
            return loss

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
            "eval_padded_relative_error_medians": evals.eval_padded_relative_error_medians,
        }
        return suite

    @staticmethod
    def get_test_eval_suite(config: dict):
        suite = {
            "eval_padded_relative_errors": evals.eval_padded_relative_errors,
            "eval_padded_relative_errors_hist": evals.eval_padded_relative_errors_hist,
        }
        return suite
