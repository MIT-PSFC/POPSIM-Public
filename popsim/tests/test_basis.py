from popsim.basis import BSplineBasis, InterpedLinearBasis
from interpax import CubicHermiteSpline
import pytest
import jax.numpy as jnp
import jax

# Helper function to check properties of the generated splines
def check_splines_properties(splines, num_splines_expected, boundary_zero=None):
    assert len(splines) == num_splines_expected, \
        f"Expected {num_splines_expected} splines, got {len(splines)}"
    for spline in splines:
        assert isinstance(spline, CubicHermiteSpline), "Each item should be a CubicHermiteSpline instance"

    # Check boundary conditions if specified
    if boundary_zero in ["left", "both"]:
        assert jnp.isclose(sum(spline(0) for spline in splines), 0, atol=1e-6), \
            "Basis should sum to zero on the left boundary"
    if boundary_zero in ["right", "both"]:
        assert jnp.isclose(sum(spline(1) for spline in splines), 0, atol=1e-6), \
            "Basis should sum to zero on the right boundary"

    # Check that the maximum value of the sum of the splines is 1
    check_vals = jnp.linspace(0, 1, 5)
    for val in check_vals:
        assert sum(spline(val) for spline in splines) <= 1, \
            "The sum of the splines should be <= 1"

# Test cases
@pytest.mark.parametrize("num_splines, boundary_zero, num_splines_expected", [
    (5, None, 5),           # No boundary constraint, should return 5 splines
    (5, "left", 5),         # Left boundary zero, should return 5 splines
    (5, "right", 5),        # Right boundary zero, should return 5 splines
    (5, "both", 5),         # Both boundaries zero, should return 5 splines
    (3, None, 3),           # Fewer splines with no boundary constraint
    (3, "left", 3),         # Fewer splines with left boundary zero
    (3, "right", 3),        # Fewer splines with right boundary zero
    (3, "both", 3),         # Fewer splines with both boundaries zero
])
def test_generate_bspline_basis_parametrized(num_splines, boundary_zero, num_splines_expected):
    splines = BSplineBasis.generate_bspline_basis(num_splines, boundary_zero)
    check_splines_properties(splines, num_splines_expected, boundary_zero)

    # Generate using BSplineBasis as well.
    basis = BSplineBasis(num_splines, boundary_zero)
    check_splines_properties(basis.basis_fns, num_splines_expected, boundary_zero)

# Edge case: smallest number of splines
def test_generate_bspline_min_splines():
    splines = BSplineBasis.generate_bspline_basis(2)
    assert len(splines) == 2, "Expected one spline when num_splines is 2"
    assert isinstance(splines[0], CubicHermiteSpline)

    with pytest.raises(ValueError):
        BSplineBasis.generate_bspline_basis(1)

# Edge case: unsupported boundary_zero argument
def test_generate_bspline_invalid_boundary():
    with pytest.raises(ValueError):
        BSplineBasis.generate_bspline_basis(5, boundary_zero="invalid")


def test_bspline_basis_internals_static():
    # Expect the data inside the BSplineBasis to be static, and hencen ot a part of the PyTree.
    basis = BSplineBasis(5)
    leaves = jax.tree.leaves(basis)
    assert len(leaves) == 0 

def test_bspline_basis():
    basis = BSplineBasis(5)
    coeffs = jnp.array([1, 2, 3, 4, 5])
    rho = jnp.linspace(0, 1, 100)
    res, vals = basis(coeffs, rho, return_components=True)
    assert res.shape == (100,)
    assert vals.shape == (5, 100)