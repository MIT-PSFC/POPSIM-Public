"""Routines to calculate the heat flux decay length (lambda_q), for several different scalings."""

import jax.numpy as np
from cfspopcon.named_options import LambdaQScaling


def calc_lambda_q(
    lambda_q_scaling: LambdaQScaling,
    average_total_pressure: float,
    power_crossing_separatrix: float,
    major_radius: float,
    B_pol_omp: float,
    inverse_aspect_ratio: float,
    magnetic_field_on_axis: float,
    q_star: float,
    lambda_q_factor: float = 1.0,
) -> float:
    """Calculate SOL heat flux decay length (lambda_q) from a scaling.

    Args:
        lambda_q_scaling: :term:`glossary link<lambda_q_scaling>`
        average_total_pressure: [atm] :term:`glossary link <average_total_pressure>`
        power_crossing_separatrix: [MW] :term:`glossary link<power_crossing_separatrix>`
        major_radius: [m] :term:`glossary link<major_radius>`
        B_pol_out_mid: [T] :term:`glossary link<B_pol_out_mid>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        q_star: [~] :term:`glossary link<q_star>`
        lambda_q_factor: [~] :term:`glossary link<lambda_q_factor>`

    Returns:
        :term:`lambda_q` [mm]
    """
    if lambda_q_scaling == LambdaQScaling.Brunner:
        return calc_lambda_q_with_brunner(
            average_total_pressure,
            lambda_q_factor=lambda_q_factor,
        )
    elif lambda_q_scaling == LambdaQScaling.EichRegression9:
        return calc_lambda_q_with_eich_regression_9(
            magnetic_field_on_axis=magnetic_field_on_axis,
            q_star=q_star,
            power_crossing_separatrix=power_crossing_separatrix,
            lambda_q_factor=lambda_q_factor,
        )
    elif lambda_q_scaling == LambdaQScaling.EichRegression14:
        return calc_lambda_q_with_eich_regression_14(
            B_pol_omp,
            lambda_q_factor=lambda_q_factor,
        )
    elif lambda_q_scaling == LambdaQScaling.EichRegression15:
        return calc_lambda_q_with_eich_regression_15(
            power_crossing_separatrix,
            major_radius,
            B_pol_omp,
            inverse_aspect_ratio,
            lambda_q_factor=lambda_q_factor,
        )
    else:
        raise NotImplementedError(f"No implementation for lambda_q scaling {lambda_q_scaling}")


def calc_lambda_q_with_brunner(average_total_pressure: float, lambda_q_factor: float = 1.0) -> float:
    """Return lambda_q according to the Brunner scaling.

    Equation 4 in :cite:`brunner_2018_heat_flux`
    """
    return lambda_q_factor * 0.91 * average_total_pressure**-0.48


def calc_lambda_q_with_eich_regression_9(
    magnetic_field_on_axis: float, q_star: float, power_crossing_separatrix: float, lambda_q_factor: float = 1.0
) -> float:
    """Return lambda_q according to Eich regression 9.

    #9 in Table 2 in :cite:`eich_scaling_2013`
    """
    return lambda_q_factor * 0.7 * magnetic_field_on_axis**-0.77 * q_star**1.05 * power_crossing_separatrix**0.09


def calc_lambda_q_with_eich_regression_14(B_pol_omp: float, lambda_q_factor: float = 1.0) -> float:
    """Return lambda_q according to Eich regression 14.

    #14 in Table 3 in :cite:`eich_scaling_2013`
    """
    return lambda_q_factor * 0.63 * B_pol_omp**-1.19


def calc_lambda_q_with_eich_regression_15(
    power_crossing_separatrix: float, major_radius: float, B_pol_omp: float, inverse_aspect_ratio: float, lambda_q_factor: float = 1.0
) -> float:
    """Return lambda_q according to Eich regression 15.

    #15 in Table 3 in :cite:`eich_scaling_2013`
    """
    lambda_q = 1.35 * major_radius**0.04 * B_pol_omp**-0.92 * inverse_aspect_ratio**0.42
    return lambda_q_factor * np.where(power_crossing_separatrix > 0, lambda_q * power_crossing_separatrix**-0.02, lambda_q)
