import jax.numpy as jnp
from jaxtyping import ArrayLike


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flattens a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The parent key to prepend. Defaults to "".
        sep (str, optional): The separator. Defaults to ".".

    Returns:
        dict: The flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = sep.join(filter(None, [parent_key, str(k)]))
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def time_epsilon(time: ArrayLike) -> ArrayLike:
    """Determine the padding amount to use for time arrays.
    Note this is better than using a fixed epsilon value as it scales the epsilon to the size of the time array.

    Args:
        time (ArrayLike): a single time value or an array of times.

    Returns:
        ArrayLike: the padding amount for each time value.
    """
    return jnp.nextafter(time, jnp.inf) - time
