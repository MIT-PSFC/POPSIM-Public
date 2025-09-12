from popsim.ml.training_config import TrainingConfig, TrainingConfig
import tempfile
import yaml
import pytest

VALID_INPUT_EXAMPLE = {
    "project": "test_project",
    "train_run_builder": "foo",
    "max_epochs": 10,
    "epochs_per_val": 2,
    "checkpoint_dir": None,
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

def test_valid_input():
    # From dictionary
    config = TrainingConfig.load(VALID_INPUT_EXAMPLE)
    assert isinstance(config, TrainingConfig)
    assert config.model_dump() == VALID_INPUT_EXAMPLE

    # From YAML file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as temp_file:
        yaml.dump(VALID_INPUT_EXAMPLE, temp_file)
        temp_file_path = temp_file.name

    config_from_yaml = TrainingConfig.load(temp_file_path)
    assert isinstance(config_from_yaml, TrainingConfig)
    assert config_from_yaml.model_dump() == VALID_INPUT_EXAMPLE

    # From module path
    config_from_module = TrainingConfig.load(__name__ + ".VALID_INPUT_EXAMPLE")
    assert isinstance(config_from_module, TrainingConfig)
    assert config_from_module.model_dump() == VALID_INPUT_EXAMPLE

def test_invalid_input():
    with pytest.raises(ValueError):
        TrainingConfig.load(INVALID_INPUT_EXAMPLE)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as temp_file:
        yaml.dump(INVALID_INPUT_EXAMPLE, temp_file)
        temp_file_path = temp_file.name

    with pytest.raises(ValueError):
        TrainingConfig.load(temp_file_path)

    with pytest.raises(ValueError):
        TrainingConfig.load(__name__ + ".INVALID_INPUT_EXAMPLE")
