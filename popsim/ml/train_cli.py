import importlib
import importlib.util

import click
import loguru

from popsim.ml import Trainer
from popsim.ml.loggers import NullLogger, WandbLogger
from popsim.ml.train_cli_config import TrainingConfigSchema, load_config_dict, load_training_config
from popsim.ml.training_run_builder import TrainRunBuilder


@click.group()
def cli():
    """CLI for training models using configuration files."""


@cli.command()
@click.argument("config_path")
@click.option("--use-wandb", is_flag=True, default=False, help="Use Weights & Biases for logging")
def launch_train(config_path: str, use_wandb: bool = False):
    """Do a training run from a configuration dictionary.

    config_path (str): Path to a yaml file or a python module path pointing to a config dict (e.g. `popsim.modules fun_module.TRAIN_CONFIG)
    use_wandb (bool): Whether to use Weights & Biases for logging. Defaults to False.
    """
    training_config = load_training_config(config_path)

    run_train(training_config, use_wandb=use_wandb)


@cli.command()
@click.argument("sweep_id")
@click.argument(
    "config_path",
)
def launch_agent(sweep_id: str, config_path: str):
    """Run a WandB agent for a sweep.

    Args:
        sweep_id (str): The ID of the WandB sweep to run.
        config_path (str): "Path to a yaml file, toml file, or a python module path pointing to a config dict (e.g. `popsim.modules.fun_module.TRAIN_CONFIG)"
    """
    training_config_schema = load_training_config(config_path)
    _agent(sweep_id, training_config_schema)


@cli.command()
@click.argument("config_path")
@click.argument("sweep_config_path")
def launch_sweep(config_path: str, sweep_config_path: str):
    """_summary_

    Args:
        config_path (str): "Path to a yaml file, toml file, or a python module path pointing to a config dict (e.g. `popsim.modules.fun_module.TRAIN_CONFIG)"
        sweep_config_path (str): "Path to a yaml file, toml file, or a python module path pointing to a sweep config dict (e.g. `popsim.modules.fun_module.SWEEP_CONFIG)"
    """
    import wandb

    training_config = load_training_config(config_path)
    sweep_config = load_config_dict(sweep_config_path)
    sweep_id = wandb.sweep(sweep_config, project=training_config.project)
    _agent(sweep_id, project=training_config.project)


def _agent(sweep_id: str, training_config_schema: TrainingConfigSchema):
    import wandb

    def train_sweep():
        return run_train(training_config_schema, use_wandb=True)

    wandb.agent(sweep_id, function=train_sweep, project=training_config_schema.project)


def get_train_run_builder_class(train_run_builder_class_path: str) -> TrainRunBuilder:
    """Get the training run builder class from a string path."""
    module_path, class_name = train_run_builder_class_path.rsplit(".", 1)
    if not importlib.util.find_spec(module_path):
        raise ImportError(f"Module {module_path} not found. Please check the config path.")
    module = importlib.import_module(module_path)
    train_run_builder_cls = getattr(module, class_name)
    if not issubclass(train_run_builder_cls, TrainRunBuilder):
        raise TypeError(f"{train_run_builder_cls} is not a subclass of TrainRunBuilder.")
    return train_run_builder_cls


def run_train(
    training_config: TrainingConfigSchema,
    use_wandb: bool = False,
):
    if use_wandb:
        import wandb

        run = wandb.init(project=training_config.project, config=training_config.model_dump())
        run.config.update({"checkpoint_dir": run.dir}, allow_val_change=True)
        logger = WandbLogger(run)
        training_config = dict(run.config)
        training_config = TrainingConfigSchema(**training_config)
    else:
        logger = NullLogger()

    train_run_builder = get_train_run_builder_class(training_config.train_run_builder_class_path)
    loguru.logger.info("Loading the dataset and creating dataloaders...")
    ds, train_dl, val_dl, test_dl = train_run_builder.get_dataloaders(training_config.dataloader_config)
    loguru.logger.info("Dataset and dataloaders created.")
    loguru.logger.info("Initializing the module...")
    model = train_run_builder.model_init(train_dl, training_config.model_init_config)
    loguru.logger.info("Initializing the loss function...")
    loss_fn = train_run_builder.get_loss_fn(training_config.loss_config)
    loguru.logger.info("Initializing the optimizer...")
    opt = train_run_builder.get_optimizer(training_config.optimizer_config)
    loguru.logger.info("Building the trainer...")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        checkpoint_dir=training_config.checkpoint_dir,
    )
    loguru.logger.info("Trainer built.")

    loguru.logger.info("Starting training...")
    test_results = trainer.train(
        train_dl,
        val_dl,
        max_epochs=training_config.max_epochs,
        epochs_per_val=training_config.epochs_per_val,
        logger=logger or NullLogger(),
        eval_suite=train_run_builder.get_val_eval_suite(training_config.val_eval_suite_config),
    )
    loguru.logger.info("Training completed.")
    return trainer, train_dl, val_dl, test_dl, test_results


if __name__ == "__main__":
    cli()
