import copy
from dataclasses import field, fields, is_dataclass

import chex
import equinox as eqx


def discrete_time_field(**kwargs) -> field:
    """Helper to construct a field with the metadata that it is a discrete time state.

    Args:
        default (Any, optional): Default value of the field. Defaults to None.

    Returns:
        field: constructed field.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["discrete_state"] = True
    return field(metadata=metadata, **kwargs)


def is_discrete_time(f: field) -> bool:
    """Check if a field is a discrete time state.

    Args:
        f (field): field to check.

    Returns:
        bool: whether the field is a discrete time state.
    """
    return f.metadata.get("discrete_state", False)


def create_discrete_state_filter_spec(obj: chex.dataclass) -> chex.dataclass:
    """Given a dataclass, create a new dataclass with the same structure,
    but with the fields replaced by whether they are discrete time states.

    Args:
        obj (chex.dataclass): dataclass to create the filter spec for.

    Returns:
        chex.dataclass: filter spec.
    """
    if not is_dataclass(obj):
        return obj

    new_obj = copy.copy(obj)
    for f in fields(obj):
        if is_dataclass(getattr(obj, f.name)):
            setattr(new_obj, f.name, create_discrete_state_filter_spec(getattr(obj, f.name)))
        else:
            setattr(new_obj, f.name, is_discrete_time(f))

    return new_obj


def partition_discrete_cont(obj: chex.dataclass) -> tuple[chex.dataclass, chex.dataclass]:
    """Partition a dataclass into discrete time and continuous time fields.

    Args:
        obj (chex.dataclass): dataclass to partition.

    Returns:
        tuple[chex.dataclass, chex.dataclass]: discrete time fields, continuous time fields.
    """
    filter_spec = create_discrete_state_filter_spec(obj)
    return eqx.partition(obj, filter_spec)
