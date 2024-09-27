import numpy as np
import popsim.array_utils as array_utils
import pytest

@pytest.mark.parametrize("arr, axis, expected", [
    # Test case 1: 1D array with contiguous True values at the end
    (
        np.array([False, False, True, True, True]),
        0,
        np.array([False, False, True, True, True])
    ),
    # Test case 2: 1D array with non-contiguous True values
    (
        np.array([False, True, False, True, True]),
        0,
        np.array([False, False, False, True, True])
    ),
    # Test case 3: 1D array with all False values
    (
        np.array([False, False, False, False]),
        0,
        np.array([False, False, False, False])
    ),
    # Test case 4: 1D array with all True values
    (
        np.array([True, True, True, True]),
        0,
        np.array([True, True, True, True])
    ),
    # Test case 5: 2D array along axis 0
    (
        np.array([
            [False, False],
            [False, True],
            [True, True]
        ]),
        0,
        np.array([
            [False, False],
            [False, True],
            [True, True]
        ])
    ),
    # Test case 6: 2D array along axis 1
    (
        np.array([
            [False, False, True],
            [False, True, True],
            [True, True, True]
        ]),
        1,
        np.array([
            [False, False, True],
            [False, True, True],
            [True, True, True]
        ])
    ),
    # Test case 7: 2D array with complex pattern along axis 1
    (
        np.array([
            [False, True, False, True, True],
            [True, True, False, False, True],
            [False, False, False, False, False]
        ]),
        1,
        np.array([
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False]
        ])
    ),
    # Test case 8: 3D array along axis 2
    (
        np.array([
            [
                [False, False, True],
                [True, True, True]
            ],
            [
                [True, False, False],
                [False, False, True]
            ]
        ]),
        2,
        np.array([
            [
                [False, False, True],
                [True, True, True]
            ],
            [
                [False, False, False],
                [False, False, True]
            ]
        ])
    ),
])
def test_contiguous_true_end_of_axis_mask(arr, axis, expected):
    result = array_utils.contiguous_true_end_of_axis_mask(arr, axis)
    np.testing.assert_array_equal(result, expected)