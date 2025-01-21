"""Threshold powers required to enter improved confinement regimes."""
import jax.numpy as np
from cfspopcon.named_options import ConfinementPowerScaling


def _calc_Martin_LH_threshold(
    magnetic_field_on_axis: float, surface_area: float, fuel_average_mass_number: float, electron_density_profile: float
) -> float:
    """See below."""
    _DEUTERIUM_MASS_NUMBER = 2.0

    return (
        0.0488
        * ((electron_density_profile / 10.0) ** 0.717)
        * (magnetic_field_on_axis**0.803)
        * (surface_area**0.941)
        * (_DEUTERIUM_MASS_NUMBER / fuel_average_mass_number)
    )


def calc_LH_transition_threshold_power(
    plasma_current: float,
    magnetic_field_on_axis: float,
    minor_radius: float,
    major_radius: float,
    surface_area: float,
    fuel_average_mass_number: float,
    average_electron_density: float,
    scale: float = 1.0,
) -> float:
    """Calculate the threshold power (crossing the separatrix) to transition into H-mode.

    From Martin NF 2008 Scaling, with mass correction :cite:`martin_power_2008`
    Added in low density branch from Ryter 2014 :cite:`Ryter_2014`

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        major_radius: [m] :term:`glossary link<major_radius>`
        surface_area: [m^2] :term:`glossary link<surface_area>`
        fuel_average_mass_number: [amu] :term:`glossary link<fuel_average_mass_number>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        scale: (optional) scaling factor for P_LH adjustment studies [~]

    Returns:
         :term:`P_LH_thresh` [MW]
    """
    neMin19 = 0.7 * (plasma_current**0.34) * (magnetic_field_on_axis**0.62) * (minor_radius**-0.95) * ((major_radius / minor_radius) ** 0.4)

    P_LH_thresh = _calc_Martin_LH_threshold(
        magnetic_field_on_axis,
        surface_area,
        fuel_average_mass_number,
        electron_density_profile=np.max(np.array([average_electron_density, neMin19])),
    )

    return np.where(
        average_electron_density < neMin19, (P_LH_thresh * (neMin19 / average_electron_density) ** 2.0) * scale, P_LH_thresh * scale
    )


def calc_LI_transition_threshold_power(plasma_current: float, average_electron_density: float, scale: float = 1.0) -> float:
    """Calculate the threshold power (crossing the separatrix) to transition into I-mode.

    Note: uses scaling described in Fig 5 of ref :cite:`hubbard_threshold_2012`

    Args:
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        scale: (optional) scaling factor for P_LI adjustment studies [~]

    Returns:
        :term:`P_LI_thresh` [MW]
    """
    return 2.11 * plasma_current**0.94 * ((average_electron_density / 10.0) ** 0.65) * scale


def calc_confinement_transition_threshold_power(
    energy_confinement_scaling: ConfinementPowerScaling,
    plasma_current: float,
    magnetic_field_on_axis: float,
    minor_radius: float,
    major_radius: float,
    surface_area: float,
    fuel_average_mass_number: float,
    average_electron_density: float,
    confinement_threshold_scalar: float = 1.0,
) -> float:
    """Calculate the threshold power (crossing the separatrix) to transition into an improved confinement mode.

    Args:
        energy_confinement_scaling: [] :term:`glossary link<energy_confinement_scaling>`
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        minor_radius: [m] :term:`glossary link<minor_radius>`
        major_radius: [m] :term:`glossary link<major_radius>`
        surface_area: [m^2] :term:`glossary link<surface_area>`
        fuel_average_mass_number: [amu] :term:`glossary link<fuel_average_mass_number>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        confinement_threshold_scalar: (optional) scaling factor for P_LH adjustment studies [~]

    Returns:
        :term:`P_LH_thresh` [MW]
    """

    if energy_confinement_scaling.name not in ["LOC", "IModey2"]:
        P_LH_thresh = calc_LH_transition_threshold_power(
            plasma_current,
            magnetic_field_on_axis,
            minor_radius,
            major_radius,
            surface_area,
            fuel_average_mass_number,
            average_electron_density,
            scale=confinement_threshold_scalar,
        )
    elif energy_confinement_scaling.name == "IModey2":
        P_LH_thresh = calc_LI_transition_threshold_power(plasma_current, average_electron_density, scale=confinement_threshold_scalar)
    else:
        raise ValueError("Encountered unhandled confinement time scaling.")

    return P_LH_thresh
