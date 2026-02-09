import importlib
import importlib.util
import inspect
import os

import loguru

from popsim.ml import DataLoader, Trainer
from popsim.ml.loggers import NullLogger, WandbLogger
from popsim.ml.train_config import TrainConfig, load_dict
from popsim.ml.train_run_builder import TrainRunBuilder


def launch_train(
    config: str | os.PathLike[str] | dict, use_wandb: bool = False
) -> tuple[Trainer, DataLoader, DataLoader, DataLoader, dict]:
    """Launch a training run from a configuration dictionary.

    Args:
        config (str | os.PathLike[str] | dict): Path to a yaml file or a python module path pointing to a config dict (e.g. `popsim.modules fun_module.TRAIN_CONFIG)
        use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.. Defaults to False.

    Returns:
        tuple[Trainer, DataLoader, DataLoader, DataLoader, dict]: Objects relevant to the training run.
    """
    training_config = TrainConfig.load(config)

    return _run_train(training_config, use_wandb=use_wandb)


def launch_sweep(config_path: str | os.PathLike[str] | dict, sweep_config_path: str | os.PathLike[str] | dict):
    """Launch a hyperparameter sweep using Weights & Biases.

    Args:
        config_path (str | os.PathLike[str] | dict): "Path to a yaml file, toml file, or a python module path pointing to a config dict (e.g. `popsim.modules.fun_module.TRAIN_CONFIG)"
        sweep_config_path (str | os.PathLike[str] | dict): "Path to a yaml file, toml file, or a python module path pointing to a sweep config dict (e.g. `popsim.modules.fun_module.SWEEP_CONFIG)"
    """
    import wandb

    training_config = TrainConfig.load(config_path)
    sweep_config = load_dict(sweep_config_path)
    sweep_id = wandb.sweep(sweep_config, project=training_config.project)
    launch_agent(config_path, sweep_id)


def launch_agent(config_path: str | os.PathLike[str] | dict, sweep_id: str):
    """Launch a Weights & Biases agent as a part of a hyperparameter sweep.

    Args:
        config_path (str | os.PathLike[str] | dict): "Path to a yaml file, toml file, or a python module path pointing to a config dict (e.g. `popsim.modules.fun_module.TRAIN_CONFIG)"
        sweep_id (str): The ID of the sweep to join.

    """
    import wandb

    training_config = TrainConfig.load(config_path)

    def _train_fn():
        return _run_train(training_config, use_wandb=True)

    wandb.agent(sweep_id, function=_train_fn, project=training_config.project)


def _get_train_run_builder_class(train_run_builder: str | os.PathLike[str] | type) -> TrainRunBuilder:
    """Get the training run builder class from a string, path, or class."""
    if inspect.isclass(train_run_builder):
        if not issubclass(train_run_builder, TrainRunBuilder):
            raise TypeError(f"Expect {train_run_builder} to be a subclass of TrainRunBuilder.")
        return train_run_builder
    module_path, class_name = train_run_builder.rsplit(".", 1)
    if not importlib.util.find_spec(module_path):
        raise ImportError(f"Module {module_path} not found. Please check the config path.")
    module = importlib.import_module(module_path)
    train_run_builder_cls = getattr(module, class_name)
    if not issubclass(train_run_builder_cls, TrainRunBuilder):
        raise TypeError(f"{train_run_builder_cls} is not a subclass of TrainRunBuilder.")
    return train_run_builder_cls


def _run_train(
    training_config: TrainConfig,
    use_wandb: bool = False,
) -> tuple[Trainer, DataLoader, DataLoader, DataLoader, dict]:
    if use_wandb:
        import wandb

        run = wandb.init(project=training_config.project, config=training_config.model_dump())
        run.config.update({"checkpoint_dir": run.dir}, allow_val_change=True)
        logger = WandbLogger(run)
        training_config = dict(run.config)
        training_config = TrainConfig(**training_config)
    else:
        logger = NullLogger()

    train_run_builder = _get_train_run_builder_class(training_config.train_run_builder)
    loguru.logger.info("Loading the dataset and creating dataloaders...")
    _, train_dl, val_dl, test_dl = train_run_builder.get_dataloaders(training_config.dataloader_config)
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
        trainable_getter=train_run_builder.get_trainable_getter(training_config.trainable_getter_config),
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
        test_dl=test_dl,
        test_eval_suite=train_run_builder.get_test_eval_suite(training_config.test_eval_suite_config),
    )
    loguru.logger.info("Training completed.")
    return trainer, train_dl, val_dl, test_dl, test_results


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "launch_train": launch_train,
            "launch_sweep": launch_sweep,
            "launch_agent": launch_agent,
        }
    )
