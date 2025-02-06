import jax
import jax.numpy as jnp
import optax

from popsim.ml import Trainer, make_standard_dataloaders
from popsim.ml.loggers import NullLogger, WandbLogger
from popsim.modules.profile_predictor.data import get_ds
from popsim.modules.profile_predictor.module import EvalEnv, ProfilePredictor


def get_dls(config):
    ds, episode_coord = get_ds(config["ds"], config["debug"])

    dls = make_standard_dataloaders(
        ds=ds,
        time_coord="time",
        episode_coord=episode_coord,
        input_vars=config["input_vars"],
        target_vars=config["target_vars"],
        extra_vars=config["extra_vars"],
        split_fracs=config["split_fracs"],
        key=config["prng_seed"],
        batch_size=config["batch_size"],
    )
    return ds, dls


def get_training_objs(config):
    ds, dls = get_dls(config)

    train_dl, val_dl, test_dl = dls

    def loss_fn(pred, targ):
        ne_rho_loss = jnp.trapezoid(
            optax.huber_loss(pred.ne.data, targ["ne20_rho"], delta=config["huber_delta"]),
            x=ds["rho"].values,
        )
        te_rho_loss = jnp.trapezoid(
            optax.huber_loss(pred.te.data, targ["Te_keV_rho"], delta=config["huber_delta"]),
            x=ds["rho"].values,
        )
        return ne_rho_loss + te_rho_loss

    prof = ProfilePredictor.init(
        dl=train_dl,
        te_shape_var="Te_shape",
        ne_shape_var="ne_shape",
        n_shapes=config["n_shapes"],
        nn_width=config["nn_width"],
        nn_depth=config["nn_depth"],
        shape_type=config["shape_type"],
        key=jax.random.PRNGKey(config["prng_seed"]),
        softmax_temp=config["softmax_temp"],
        use_ne_edge=config["use_ne_edge"],
    )

    env = EvalEnv(prof, ds["rho"].values)

    schedule = optax.exponential_decay(
        init_value=config["lr0"],
        transition_steps=config["transition_steps"],
        decay_rate=config["decay_rate"],
        end_value=config["lrf"],
    )

    opt = optax.adamw(learning_rate=schedule, weight_decay=config["weight_decay"])

    trainer = Trainer(
        model=env,
        loss_fn=loss_fn,
        optimizer=opt,
        trainable_getter=lambda m: m.get_trainable(freeze_shapes=config["freeze_shapes"]),
    )
    return trainer, train_dl, val_dl, test_dl


def train(config: dict):
    if config["use_wandb"]:
        import wandb

        run = wandb.init(project=config["sparc_project"], config=config)
        logger = WandbLogger(run)
        config = run.config
    else:
        logger = NullLogger()

    trainer, train_dl, val_dl, test_dl = get_training_objs(config)

    trainer.train(
        train_dl,
        val_dl,
        max_epochs=config["max_epochs"],
        epochs_per_val=config["epochs_per_val"],
        logger=logger,
        eval_suite=config["train_eval_suite"],
    )
    return trainer, train_dl, val_dl, test_dl
