import numpy as np
import popsim.array_utils as array_utils
import pytest
import jax.numpy as jnp

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


@pytest.mark.parametrize(
    "ys, expected_output",
    [
        # Test case 1: Regular array with NaNs
        (jnp.array([1.0, jnp.nan, 3.0, jnp.nan, 5.0]), jnp.array([1.0, 1.0, 3.0, 3.0, 5.0])),
        # Test case 2: Array with NaNs at the beginning
        (jnp.array([jnp.nan, 2.0, 3.0, 4.0, 5.0]), jnp.array([jnp.nan, 2.0, 3.0, 4.0, 5.0])),
        # Test case 3: Array with NaNs at the end
        (jnp.array([1.0, 2.0, 3.0, 4.0, jnp.nan]), jnp.array([1.0, 2.0, 3.0, 4.0, 4.0])),
        # Test case 4: Array with consecutive NaNs
        (jnp.array([1.0, jnp.nan, jnp.nan, 4.0, 5.0]), jnp.array([1.0, 1.0, 1.0, 4.0, 5.0])),
        # Test case 5: Array with all NaNs
        (jnp.array([jnp.nan, jnp.nan, jnp.nan]), jnp.array([jnp.nan, jnp.nan, jnp.nan])),
        # Test case 6: Array with no NaNs
        (jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0])),
        # Test case 7: Empty array
        (jnp.array([]), jnp.array([])),
    ]
)
def test_forward_fill_nans(ys, expected_output):
    result = array_utils.forward_fill_nans(ys)
    assert jnp.allclose(result, expected_output, equal_nan=True)

@pytest.mark.parametrize(
    "ys",
    [
        # Test case 1: Non-1D array
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        # Test case 2: Scalar input
        jnp.array(1.0),
    ]
)
def test_forward_fill_nans_invalid_input(ys):
    with pytest.raises(ValueError):
        array_utils.forward_fill_nans(ys)