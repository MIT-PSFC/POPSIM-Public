import inspect


def print_class_without_methods(cls, ignored_methods: set[str]) -> str:
    """Print a class without specified methods.
    Intended for use in notebooks to show the state/inputs/outputs of a module
    without cluttering the display with method definitions.

    Args:
        cls: The class to print.
        ignored_methods: A set of method names to ignore.

    Returns:
        The source code of the class as a string, excluding the specified methods.
    """
    source_lines = inspect.getsource(cls).splitlines()

    filtered_lines = []
    skip_until_dedent = False

    for line in source_lines:
        # Check if this line starts a method we want to ignore
        if any(f"def {method}" in line for method in ignored_methods):
            skip_until_dedent = True
            continue

        # If skipping, check if we've returned to class level (dedented)
        if skip_until_dedent:
            if (line and not line[0].isspace()) or (line.strip().startswith("def ") and not any(m in line for m in ignored_methods)):
                skip_until_dedent = False
            else:
                continue

        filtered_lines.append(line)

    return chr(10).join(filtered_lines)
