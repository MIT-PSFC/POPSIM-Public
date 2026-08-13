import pytest

import popsim.utils as utils


@pytest.mark.parametrize(
    "d, parent_key, sep, expected",
    [
        # Test case 1: Simple flat dictionary
        ({"a": 1, "b": 2}, "", ".", {"a": 1, "b": 2}),
        # Test case 2: Nested dictionary
        ({"a": {"b": 2}}, "", ".", {"a.b": 2}),
        # Test case 3: Deeply nested dictionary
        ({"a": {"b": {"c": 3}}}, "", ".", {"a.b.c": 3}),
        # Test case 4: With parent key
        ({"b": 2}, "a", ".", {"a.b": 2}),
        # Test case 5: Mixed types
        ({"a": {"b": 2, "c": [1, 2]}, "d": 4}, "", ".", {"a.b": 2, "a.c": [1, 2], "d": 4}),
        # Test case 6: With different separator
        ({"a": {"b": 2}}, "", "/", {"a/b": 2}),
        # Test case 7: Empty dictionary
        ({}, "", ".", {}),
    ]
)
def test_flatten_dict(d, parent_key, sep, expected):
    result = utils.flatten_dict(d, parent_key=parent_key, sep=sep)
    assert result == expected