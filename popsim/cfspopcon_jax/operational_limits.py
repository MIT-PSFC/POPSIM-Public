"""Operational limits to avoid disruptive regions."""
import jax.numpy as np


def calc_greenwald_fraction(
    average_electron_density: float, inverse_aspect_ratio: float, major_radius: float, plasma_current: float
) -> float:
    """Calculate the fraction of the Greenwald density limit.

    Args:
        average_electron_density: [1e20 m^-3] :term:`glossary link<average_electron_density>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        major_radius: [m] :term:`glossary link<major_radius>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`

    Returns:
        :term:`greenwald_fraction` [~]
    """
    n_Greenwald = calc_greenwald_density_limit(plasma_current, inverse_aspect_ratio * major_radius)

    return average_electron_density / n_Greenwald


def calc_greenwald_density_limit(plasma_current: float, minor_radius: float) -> float:
    """Calculate the Greenwald density limit.

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        minor_radius: [m] :term:`glossary link<minor_radius>`

    Returns:
        nG Greenwald density limit [n20]
    """
    return plasma_current / (np.pi * minor_radius**2)


def calc_troyon_limit(minor_radius: float, magnetic_field_on_axis: float, plasma_current: float) -> float:
    """Calculate the maximum value for beta, according to the Troyon limit.

    Args:
        minor_radius: [m] :term:`glossary link<minor_radius>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`

    Returns:
         troyon_max_beta [~]
    """
    return 2.8 * plasma_current / (minor_radius * magnetic_field_on_axis)
