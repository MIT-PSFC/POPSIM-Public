import tempfile

import pytest
import yaml
from pydantic import ValidationError

from popsim.ml.train_config import TrainConfig

VALID_INPUT_EXAMPLE = {
    "project": "test_project",
    "train_run_builder": "foo",
    "max_epochs": 10,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
    "patience": None,
    "dataloader_config": {},
    "model_init_config": {},
    "loss_config": {},
    "optimizer_config": {},
    "trainable_getter_config": None,
    "val_eval_suite_config": None,
    "test_eval_suite_config": None,
}

INVALID_INPUT_EXAMPLE = {
    "foo": "bar",
}

# Fields not in VALID_INPUT_EXAMPLE take their defaults on load.
EXPECTED_DUMP = VALID_INPUT_EXAMPLE | {
    "resume": False,
    "max_wall_seconds": None,
}

def test_valid_input():
    # From dictionary
    config = TrainConfig.load(VALID_INPUT_EXAMPLE)
    assert isinstance(config, TrainConfig)
    assert config.model_dump() == EXPECTED_DUMP

    # From YAML file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as temp_file:
        yaml.dump(VALID_INPUT_EXAMPLE, temp_file)
        temp_file_path = temp_file.name

    config_from_yaml = TrainConfig.load(temp_file_path)
    assert isinstance(config_from_yaml, TrainConfig)
    assert config_from_yaml.model_dump() == EXPECTED_DUMP

    # From module path
    config_from_module = TrainConfig.load(__name__ + ".VALID_INPUT_EXAMPLE")
    assert isinstance(config_from_module, TrainConfig)
    assert config_from_module.model_dump() == EXPECTED_DUMP

def test_invalid_input():
    with pytest.raises(ValueError):
        TrainConfig.load(INVALID_INPUT_EXAMPLE)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as temp_file:
        yaml.dump(INVALID_INPUT_EXAMPLE, temp_file)
        temp_file_path = temp_file.name

    with pytest.raises(ValueError):
        TrainConfig.load(temp_file_path)

    with pytest.raises(ValueError):
        TrainConfig.load(__name__ + ".INVALID_INPUT_EXAMPLE")

def test_isolated_configs():
    config_a = TrainConfig.load(VALID_INPUT_EXAMPLE)

    with pytest.raises(ValidationError):
        config_a.project = "modified_project"

    config_b = config_a.model_copy()
    assert config_a == config_b

    config_a.dataloader_config.update({"new_key": "new_value"})
    assert config_b.dataloader_config.get("new_key") is None

    with pytest.raises(Warning):
        config_a.model_copy(deep=False)
