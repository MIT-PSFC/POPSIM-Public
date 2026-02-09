import jax
import optax
import xarray as xr

from popsim.math_utils import padded_relative_error
from popsim.ml import DataLoader, TrainRunBuilder, make_standard_dataloaders
from popsim.modules.meqml.fast_obs import evals
from popsim.modules.meqml.fast_obs.module import FastObs


class FastObsTrainRunBuilder(TrainRunBuilder):
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
    def model_init(train_dl: DataLoader, model_init_config: dict) -> FastObs:
        """
        Instantiate and return your model given a training DataLoader
        and a model config dict.
        """

        module = FastObs.init(train_dl, **model_init_config)
        return module

    @staticmethod
    def get_loss_fn(config: dict):
        def loss_fn(pred, targ):
            errors = {
                "Ip": padded_relative_error(pred.Ip, targ["LY.Ip"].data),
                "rIp": padded_relative_error(pred.rIp, targ["LY.rIp"].data),
                "zIp": padded_relative_error(pred.zIp, targ["LY.zIp"].data),
            }

            huber_losses = jax.tree.map(lambda x: optax.huber_loss(x, delta=config["huber_delta"]), errors)

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
            "mean_relative_error": evals.eval_relative_error_means,
        }
        return suite

    @staticmethod
    def get_test_eval_suite(config: dict):
        suite = {
            "eval_padded_relative_errors": evals.eval_padded_relative_errors,
            "padded_relative_error_hist": evals.padded_relative_error_hist,
        }
        return suite
