from typing import Optional, Protocol

import equinox as eqx
import jax.numpy as jnp
from interpax import CubicHermiteSpline
from jaxtyping import Array


class Basis1DProtocol(Protocol):
    """Protocol for a basis function on a 1D grid."""

    def __call__(self, coeffs: Array, grid: Array) -> Array:
        """Evaluate the basis with given coefficients at the given grid points.

        Args:
            coeffs (Array): Coefficients corresponding to each basis function.
            grid (Array): Grid points at which to evaluate the basis functions.

        Returns:
            Array: The result of combining the basis functions with the coefficients.
        """
        ...

    @property
    def n_basis(self) -> int:
        """Number of basis functions."""
        ...


class BSplineBasis(eqx.Module):
    basis_fns: list[CubicHermiteSpline] = eqx.field(static=True)

    def __init__(self, n_comps: int, boundary_zero: Optional[str] = None):
        self.basis_fns = self.generate_bspline_basis(n_comps, boundary_zero)

    def __call__(self, coeffs: Array, rho: Array, return_components: bool = False) -> Array:
        """Evaluate the basis components at the given coefficients and rho values.

        Args:
            coeffs (Array): coefficients where the ith element corresponds to the ith basis function.
            rho (Array): grid values at which to evaluate the basis functions.
            return_components (bool, optional): Whether to return the individual components as a second argument. Defaults to False.

        Returns:
            Array: an array with dimensions (n_comps, len(rho)) containing the basis function values at each rho value. If return_components is True, the individual components are also returned.
        """
        basis_values = jnp.stack([fn(rho) for fn in self.basis_fns], axis=0)

        components = jnp.multiply(basis_values, coeffs[:, None])

        result = jnp.sum(components, axis=0)

        if return_components:
            return result, components
        else:
            return result

    @property
    def n_basis(self) -> int:
        return len(self.basis_fns)

    @staticmethod
    def generate_bspline_basis(n_comps: int, boundary_zero: Optional[str] = None) -> list[CubicHermiteSpline]:
        """Generate a basis of cubic B-splines on the interval [0, 1] using Cubic Hermite splines.

        Args:
            n_comps (int): Number of splines to generate for the basis. Specifies the granularity of the spline space.
            boundary_zero (Optional[str], optional): Determines whether the basis set should sum to zero on one or both boundaries.
                Options are:
                - "left": Ensures the sum of the basis is zero on the left boundary.
                - "right": Ensures the sum of the basis is zero on the right boundary.
                - "both": Ensures the sum of the basis is zero on both boundaries.
                Defaults to None, in which case no boundary constraint is applied.

        Returns:
            list[CubicHermiteSpline]: A list of `CubicHermiteSpline` objects forming the B-spline basis.
        """

        if n_comps < 2:
            raise ValueError("Number of splines must be at least 2")
        if boundary_zero not in ["left", "right", "both", None]:
            raise ValueError(f"Invalid boundary_zero option: {boundary_zero}")

        # We handle boundary_zero by simply adding more knot points and removing the extra splines.
        if boundary_zero in ["left", "right"]:
            n_comps += 1
        elif boundary_zero == "both":
            n_comps += 2

        x = jnp.linspace(0, 1, n_comps)

        splines = []
        for i in range(n_comps):
            # Define interpolation points for CubicHermiteSpline
            y = jnp.zeros_like(x)
            dydx = jnp.zeros_like(x)

            y = y.at[i].set(1.0)
            dydx = dydx.at[i].set(0.0)

            spline = CubicHermiteSpline(x=x, y=y, dydx=dydx, extrapolate=False)
            splines.append(spline)

        # Handle boundary_zero by removing the first or last spline.
        if boundary_zero == "left":
            return splines[1:]
        elif boundary_zero == "right":
            return splines[:-1]
        elif boundary_zero == "both":
            return splines[1:-1]
        else:
            return splines


class InterpedLinearBasis(eqx.Module):
    grid: Array = eqx.field(static=True)

    def __init__(self, grid: Array):
        self.grid = grid

    def __call__(self, coeffs: Array, rho: Array) -> Array:
        result = jnp.interp(rho, self.grid, coeffs)
        return result
