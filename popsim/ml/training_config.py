import importlib
import importlib.util
import os
from typing import Optional, Union

import yaml
from pydantic import BaseModel

from popsim.ml.training_run_builder import TrainRunBuilder


class TrainingConfig(BaseModel):
    project: str
    train_run_builder: Union[str, type[TrainRunBuilder]]
    max_epochs: int
    epochs_per_val: int
    checkpoint_dir: Optional[str] = None
    dataloader_config: dict
    model_init_config: dict
    loss_config: dict
    optimizer_config: dict
    trainable_getter_config: Optional[dict] = None
    val_eval_suite_config: Optional[dict] = None
    test_eval_suite_config: Optional[dict] = None

    @classmethod
    def load(cls, config_or_path: str | os.PathLike[str] | dict) -> "TrainingConfig":
        """Load a configuration from a yaml file, a Python module path, or directly from a dictionary.

        Example usages:
            config: TrainingConfig = TrainingConfig.load("path/to/config.yaml")
            config: TrainingConfig = TrainingConfig.load("popsim.modules.fun_module.TRAIN_CONFIG")
            config: TrainingConfig = TrainingConfig.load({"project": "test", ...})

        Args:
            config_or_path (str | os.PathLike[str] | dict): Path to the configuration file or module, or the config dictionary itself.
                Can be a string path, PathLike object, or dictionary.

        Raises:
            ImportError: If the specified module path cannot be found.
            ValidationError: If the loaded configuration doesn't match the schema.

        Returns:
            The loaded configuration as a TrainingConfig instance.
        """
        config_or_path = load_dict(config_or_path)
        out = cls(**config_or_path)
        return out


def load_dict(config_or_path: str | os.PathLike[str] | dict) -> dict:
    """Load a configuration dictionary, given an input that can be a path to a YAML file,
    a Python module path, or the dictionary itself.

    Args:
        config_or_path (str | os.PathLike[str] | dict): Path to a YAML file, a Python module path (e.g. 'popsim.modules.fun_module.TRAIN_CONFIG'), or the config dictionary itself.

    Raises:
        ImportError: If the specified module path cannot be found.

    Returns:
        dict: The loaded configuration dictionary.
    """
    if isinstance(config_or_path, dict):
        return config_or_path

    # Load the configuration dictionary.
    if config_or_path.endswith((".yaml", ".yml")):
        with open(config_or_path) as f:
            config_or_path = yaml.safe_load(f)
    else:
        module_path, dict_name = config_or_path.rsplit(".", 1)
        # Check if the module can be imported.
        if not importlib.util.find_spec(module_path):
            raise ImportError(f"Module {module_path} not found. Please check the config path.")
        config_or_path = importlib.import_module(module_path).__dict__[dict_name]
    return config_or_path
