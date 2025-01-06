from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Callable

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


def no_save_field(**kwargs) -> field:
    """Helper to construct a field with the metadata that it should not be saved.

    Args:
        default (Any, optional): Default value of the field. Defaults to None.

    Returns:
        field: constructed field.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["no_save"] = True
    return field(metadata=metadata, **kwargs)


def discrete_no_save_field(**kwargs) -> field:
    """Helper to construct a field with both discrete time and no_save metadata.

    Args:
        default (Any, optional): Default value of the field. Defaults to None.

    Returns:
        field: constructed field with both discrete_state and no_save metadata.
    """
    metadata = kwargs.pop("metadata", {})
    metadata["discrete_state"] = True
    metadata["no_save"] = True
    return field(metadata=metadata, **kwargs)


def is_discrete_time(f: field) -> bool:
    """Check if a field is a discrete time state.

    Args:
        f (field): field to check.

    Returns:
        bool: whether the field is a discrete time state.
    """
    return f.metadata.get("discrete_state", False)


def is_no_save(f: field) -> bool:
    """Check if a field is marked as no_save.

    Args:
        f (field): field to check.

    Returns:
        bool: whether the field is marked as no_save.
    """
    return f.metadata.get("no_save", False)


def create_filter_spec(obj: dataclass, field_condition: Callable[[field], bool]) -> dataclass:
    """Given a dataclass, create a new dataclass with the same structure,
    but with the fields replaced by whether they satisfy a specified condition.

    Args:
        obj (dataclass): dataclass to create the filter spec for.
        field_condition (Callable[[field], bool]): Function that defines the condition for each field.

    Returns:
        dataclass: Filter spec based on the specified condition.
    """
    if not is_dataclass(obj):
        return obj

    def is_static_field(f: field) -> bool:
        return f.metadata.get("static", False)

    non_static_fields = [f for f in fields(obj) if not is_static_field(f)]

    field_filter_spec = {
        f.name: create_filter_spec(getattr(obj, f.name), field_condition) if is_dataclass(getattr(obj, f.name)) else field_condition(f)
        for f in non_static_fields
    }

    return replace(
        obj,
        **field_filter_spec,
    )


def partition_discrete_cont(obj: chex.dataclass) -> tuple[chex.dataclass, chex.dataclass]:
    """Partition a dataclass into discrete time and continuous time fields.

    Args:
        obj (chex.dataclass): dataclass to partition.

    Returns:
        tuple[chex.dataclass, chex.dataclass]: discrete time fields, continuous time fields.
    """
    filter_spec = create_filter_spec(obj, is_discrete_time)
    return eqx.partition(obj, filter_spec)


def partition_save_no_save(obj: chex.dataclass) -> tuple[chex.dataclass, chex.dataclass]:
    """Partition a dataclass into fields that should be saved and those that should not be saved.

    Args:
        obj (chex.dataclass): dataclass to partition.

    Returns:
        tuple[chex.dataclass, chex.dataclass]: fields that should be saved, fields that should not be saved.
    """
    filter_spec = create_filter_spec(obj, is_no_save)
    no_save, save = eqx.partition(obj, filter_spec)
    return save, no_save
