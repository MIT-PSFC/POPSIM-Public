import importlib
import importlib.util
from typing import Optional, Union

import yaml
from pydantic import BaseModel


class TrainingConfigSchema(BaseModel):
    project: str
    training_spec_class_path: str
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


def load_config_dict(config_or_path: Union[str, dict]) -> dict:
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


def load_training_config(config_or_path: Union[str, dict]) -> TrainingConfigSchema:
    """Load a configuration from a yaml file, a Python module path, or directly from a dictionary.

    Example usages:
        config = load_training_config("path/to/config.yaml")
        config = load_training_config("popsim.modules.fun_module.TRAIN_CONFIG")
        config = load_training_config({"project": "test", ...})

    Args:
        config (Union[str, dict]): Path to the configuration file or module or the config dictionary itself.

    Raises:
        ValueError: If the loaded configuration is not a dictionary.

    Returns:
        TrainingConfigSchema: The loaded configuration as a TrainingConfigSchema instance.
    """
    config_or_path = load_config_dict(config_or_path)
    out = TrainingConfigSchema(**config_or_path)
    return out
