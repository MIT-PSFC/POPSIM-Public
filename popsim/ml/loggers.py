import yaml


def flatten_dict(dictionary, parent_key="", sep="/"):
    """Flatten a nested dictionary, using a separator for nested keys."""
    items = {}
    for k, v in dictionary.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, parent_key=new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def simplified_repr(dictionary, max_string_size):
    """Return a simplified representation of a dictionary, truncating strings and replacing most objects with their type."""
    truncated_dict = {}
    for key, val in dictionary.items():
        # If the value is a dictionary, recurse
        if isinstance(val, dict):
            truncated_dict[key] = simplified_repr(val, max_string_size)
        else:
            str_val = str(val)
            if len(str_val) <= max_string_size:
                truncated_dict[key] = val
            else:
                val_type = str(type(val)).split("'")[1]
                truncated_dict[key] = f"{val_type}: {str_val[:max_string_size]}..."
    return truncated_dict


class LoggerBase:
    def __init__(self):
        pass

    def log(self, dictionary):
        raise NotImplementedError


class NullLogger(LoggerBase):
    def log(self, dictionary):
        return


class ConsoleLogger(LoggerBase):
    def __init__(self, max_string_size=50):
        self.max_string_size = max_string_size

    def log(self, dictionary):
        truncated_dict = simplified_repr(dictionary, self.max_string_size)
        print(yaml.dump(truncated_dict, default_flow_style=False))


class WandbLogger(LoggerBase):
    def __init__(self, run):
        self.run = run
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("val/*", step_metric="val/epoch")

    def log(self, dictionary):
        self.run.log(flatten_dict(dictionary))


class TrainerLoggingInterface:
    logger: LoggerBase

    def __init__(self, logger: LoggerBase):
        self.logger = logger

    def log_train_step(self, dictionary, step: int):
        log_dict = {f"train/{key}": val for key, val in dictionary.items()}
        log_dict.update({"train/step": step})
        self.log(log_dict)

    def log_train_epoch(self, dictionary, epoch: int):
        log_dict = {f"train/{key}": val for key, val in dictionary.items()}
        log_dict.update({"epoch": epoch})
        self.log(log_dict)

    def log_validation(self, dictionary, epoch: int):
        log_dict = {f"val/{key}": val for key, val in dictionary.items()}
        log_dict.update({"epoch": epoch})
        self.log(log_dict)

    def log_test(self, dictionary):
        self.log({f"test/{key}": val for key, val in dictionary.items()})


def get_logger(logger_type: str, **kwargs) -> LoggerBase:
    """
    Factory function to create a logger based on the specified type.

    Args:
        logger_type (str): The type of logger to create ('null', 'console', or 'wandb').
        **kwargs: Additional arguments to pass to the logger constructor.

    Returns:
        LoggerBase: An instance of the specified logger type.

    Raises:
        ValueError: If an invalid logger type is specified.
    """
    if logger_type == "null":
        return NullLogger()
    elif logger_type == "console":
        return ConsoleLogger(**kwargs)
    elif logger_type == "wandb":
        return WandbLogger(**kwargs)
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")
