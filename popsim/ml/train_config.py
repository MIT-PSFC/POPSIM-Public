import importlib
import importlib.util
import os

import yaml
from pydantic import BaseModel

from popsim.ml.train_run_builder import TrainRunBuilder


class TrainConfig(BaseModel):
    project: str  # The name of the project, primarily for logging purposes.
    train_run_builder: str | type[TrainRunBuilder]  # A string path to the TrainRunBuilder class or the class itself.
    max_epochs: int  # Maximum number of training epochs.
    epochs_per_val: int  # Number of epochs between each validation.
    checkpoint_dir: str | None = None  # Directory to save model checkpoints. If None, checkpoints are not saved.
    dataloader_config: dict  # Configuration dictionary for building the dataloaders.
    model_init_config: dict  # Configuration dictionary for initializing the model.
    loss_config: dict  # Configuration dictionary used to initialize the loss function.
    optimizer_config: dict  # Configuration dictionary used to initialize the optimizer.
    trainable_getter_config: dict | None = None  # Configuration dictionary used in the function that gets trainable parameters.
    val_eval_suite_config: dict | None = None  # Configuration dictionary used to build the validation evaluation suite.
    test_eval_suite_config: dict | None = None  # Configuration dictionary used to build the test evaluation suite.

    class Config:
        frozen = True  # Make the model immutable after creation.

    def model_copy(self, *, update=None, deep=True, **kwargs):
        if not deep:
            raise Warning("Shallow copy of TrainConfig instances are forbidden to prevent side effects from mutable fields.")
        return super().model_copy(update=update, deep=True, **kwargs)

    @classmethod
    def load(cls, config_or_path: str | os.PathLike[str] | dict) -> "TrainConfig":
        """Load a configuration from a yaml file, a Python module path, or directly from a dictionary.

        Example usages:
            config: TrainConfig = TrainConfig.load("path/to/config.yaml")
            config: TrainConfig = TrainConfig.load("popsim.modules.fun_module.TRAIN_CONFIG")
            config: TrainConfig = TrainConfig.load({"project": "test", ...})

        Args:
            config_or_path (str | os.PathLike[str] | dict): Path to the configuration file or module, or the config dictionary itself.
                Can be a string path, PathLike object, or dictionary.

        Raises:
            ImportError: If the specified module path cannot be found.
            ValidationError: If the loaded configuration doesn't match the schema.

        Returns:
            The loaded configuration as a TrainConfig instance.
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
