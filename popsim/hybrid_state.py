from dataclasses import dataclass, field, fields, is_dataclass, replace

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


def create_discrete_state_filter_spec(obj: dataclass) -> dataclass:
    """Given a dataclass, create a new dataclass with the same structure,
    but with the fields replaced by whether they are discrete time states.

    Args:
        obj (dataclass): dataclass to create the filter spec for.

    Returns:
        dataclass: filter spec.
    """
    if not is_dataclass(obj):
        return obj

    def is_static_field(f: field) -> bool:
        return f.metadata.get("static", False)

    non_static_fields = [f for f in fields(obj) if not is_static_field(f)]

    field_filter_spec = {
        f.name: create_discrete_state_filter_spec(getattr(obj, f.name)) if is_dataclass(getattr(obj, f.name)) else is_discrete_time(f)
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
    filter_spec = create_discrete_state_filter_spec(obj)
    return eqx.partition(obj, filter_spec)
