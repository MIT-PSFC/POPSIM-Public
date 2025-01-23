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
