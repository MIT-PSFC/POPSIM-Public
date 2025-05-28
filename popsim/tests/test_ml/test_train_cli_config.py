from popsim.ml.train_cli_config import load_training_config, TrainingConfigSchema
import tempfile
import yaml
import pytest

VALID_INPUT_EXAMPLE = {
    "project": "test_project",
    "training_spec_class_path": "foo",
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
    config = load_training_config(VALID_INPUT_EXAMPLE)
    assert isinstance(config, TrainingConfigSchema)
    assert config.model_dump() == VALID_INPUT_EXAMPLE

    # From YAML file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as temp_file:
        yaml.dump(VALID_INPUT_EXAMPLE, temp_file)
        temp_file_path = temp_file.name

    config_from_yaml = load_training_config(temp_file_path)
    assert isinstance(config_from_yaml, TrainingConfigSchema)
    assert config_from_yaml.model_dump() == VALID_INPUT_EXAMPLE

    # From module path
    config_from_module = load_training_config(__name__ + ".VALID_INPUT_EXAMPLE")
    assert isinstance(config_from_module, TrainingConfigSchema)
    assert config_from_module.model_dump() == VALID_INPUT_EXAMPLE

def test_invalid_input():
    with pytest.raises(ValueError):
        load_training_config(INVALID_INPUT_EXAMPLE)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml", encoding="utf-8") as temp_file:
        yaml.dump(INVALID_INPUT_EXAMPLE, temp_file)
        temp_file_path = temp_file.name

    with pytest.raises(ValueError):
        load_training_config(temp_file_path)

    with pytest.raises(ValueError):
        load_training_config(__name__ + ".INVALID_INPUT_EXAMPLE")
