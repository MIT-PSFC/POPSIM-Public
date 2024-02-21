import diffrax


def cubic_interp(ts, tree) -> diffrax.CubicInterpolation:
    return diffrax.CubicInterpolation(ts=ts, coeffs=diffrax.backward_hermite_coefficients(ts, tree))
