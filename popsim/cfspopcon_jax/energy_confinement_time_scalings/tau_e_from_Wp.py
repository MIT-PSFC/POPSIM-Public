from pathlib import Path

import jax
import yaml
from cfspopcon.named_options import ConfinementScaling
from cfspopcon.np_variant import np

# Preload the scalings (instead of doing fileio in loop)
with open(Path(__file__).parent / "tau_e_scalings.yaml") as f:
    TAU_E_SCALINGS = yaml.safe_load(f)


def get_calc_tau_e_and_P_in_from_scaling(scaling: ConfinementScaling):
    scaling_data = TAU_E_SCALINGS[scaling.name]["params"]

    def fn(
        confinement_time_scalar: float,
        plasma_current: float,
        magnetic_field_on_axis: float,
        average_electron_density: float,
        major_radius: float,
        areal_elongation: float,
        separatrix_elongation: float,
        inverse_aspect_ratio: float,
        fuel_average_mass_number: float,
        triangularity_psi95: float,
        separatrix_triangularity: float,
        plasma_stored_energy: float,
        q_star: float,
    ):
        return calc_tau_e_and_P_in_from_scaling(
            confinement_time_scalar=confinement_time_scalar,
            plasma_current=plasma_current,
            magnetic_field_on_axis=magnetic_field_on_axis,
            average_electron_density=average_electron_density,
            major_radius=major_radius,
            areal_elongation=areal_elongation,
            separatrix_elongation=separatrix_elongation,
            inverse_aspect_ratio=inverse_aspect_ratio,
            fuel_average_mass_number=fuel_average_mass_number,
            triangularity_psi95=triangularity_psi95,
            separatrix_triangularity=separatrix_triangularity,
            plasma_stored_energy=plasma_stored_energy,
            q_star=q_star,
            scaling=scaling_data,
        )

    return fn


def calc_tau_e_and_P_in_from_scaling(
    confinement_time_scalar: float,
    plasma_current: float,
    magnetic_field_on_axis: float,
    average_electron_density: float,
    major_radius: float,
    areal_elongation: float,
    separatrix_elongation: float,
    inverse_aspect_ratio: float,
    fuel_average_mass_number: float,
    triangularity_psi95: float,
    separatrix_triangularity: float,
    plasma_stored_energy: float,
    q_star: float,
    scaling: dict[str, float],
) -> tuple[float, float]:
    r"""Calculate energy confinement time and input power from a tau_E scaling.

    The energy confinement time can generally be written as

    .. math::
        \tau_e = H \cdot C \cdot P_{\tau}^{\alpha_P}
        \cdot I_{MA}^{\alpha_I} \cdot B_0^{\alpha_B} \cdot \bar{n_{e,19}}^{\alpha_n} \cdot R_0^{\alpha_R}
        \cdot \kappa_A^{\alpha_{ka}} \cdot \kappa_{sep}^{\alpha_{ks}} \cdot \epsilon^{\alpha_e}
        \cdot m_i^{\alpha_A} \cdot \delta^{\alpha_d}

    We don't know :math:`P_{\tau}` in advance, so instead write

    .. math:: \tau_e = \gamma \cdot P_{\tau}^{\alpha_P}

    with

    .. math::
        \gamma = H \cdot C
        \cdot I_{MA}^{\alpha_I} \cdot B_0^{\alpha_B} \cdot \bar{n_{e,19}}^{\alpha_n} \cdot R_0^{\alpha_R}
        \cdot \kappa_A^{\alpha_{ka}} \cdot \kappa_{sep}^{\alpha_{ks}} \cdot \epsilon^{\alpha_e}
        \cdot m_i^{\alpha_A} \cdot \delta^{\alpha_d}

    We have everything that we need to evaluate :math:`\gamma`. Then, we also know that

    .. math:: \tau_e = W_p / P_{loss}

    Then, we crucially need to define what exactly we mean by the two powers that we've introduced
    (:math:`P_{\tau}` and :math:`P_{loss}`). We usually take
    :math:`P_{\tau} = P_{heating} = P_{ohmic} + P_{\alpha} + P_{aux}` and
    :math:`P_{loss}=P_{SOL} + P_{rad,core}` [Wesson definition]. From a
    simple power balance, :math:`P_{heating}=P_{loss}` and so, setting the two :math:`\tau_e` equations equal, we get that

    .. math::
        W_p / P = \gamma \cdot P^{\alpha_P}
        P^{\alpha_P + 1} = W_p / \gamma
        P = \left(W_p / \gamma \right)^{\frac{1}{\alpha_P + 1}}

    Once we have :math:`P = P_{ohmic} + P_{\alpha} + P_{aux} = P_{SOL} + P_{rad,core}`, we can calculate :math:`W_p/P`.

    However, it is also possible that the core radiated power is subtracted when calculating :math:`\tau_e`
    [Freidberg definition of :math:`W_p`], giving

    .. math::
        P_{\tau} = P_{ohmic} + P_{\alpha} + P_{aux} - P_{rad} = P_{loss} = P_{SOL}

    If you are using a scaling where this is the case, set ``tau_e_scaling_uses_P_in=False``.
    Then, the returned value should be interpreted as :math:`P_{SOL}`.

    N.b. there are two more possible cases, where different powers are used in the two :math:`\tau_e` scalings.
    We don't allow these cases, since 1) experiments generally pick a consistent definition for :math:`P`
    and 2) this results in an equation for :math:`P` which cannot be solved analytically.

    Args:
        confinement_time_scalar: [~] confinement scaling factor
        plasma_current: [MA] :term:`glossary link<plasma_current>`
        magnetic_field_on_axis: [T] :term:`glossary link<magnetic_field_on_axis>`
        average_electron_density: [1e19 m^-3] :term:`glossary link<average_electron_density>`
        major_radius: [m] :term:`glossary link<major_radius>`
        areal_elongation: [~] :term:`glossary link<areal_elongation>`
        separatrix_elongation: [~] :term:`glossary link<separatrix_elongation>`
        inverse_aspect_ratio: [~] :term:`glossary link<inverse_aspect_ratio>`
        fuel_average_mass_number: [amu] :term:`glossary link<fuel_average_mass_number>`
        triangularity_psi95: [~] :term:`glossary link<triangularity_psi95>`
        separatrix_triangularity: [~] :term:`glossary link<separatrix_triangularity>`
        plasma_stored_energy: [MJ] :term:`glossary link<plasma_stored_energy>`
        q_star: [~] :term:`glossary link<q_star>`
        tau_e_scaling: [] :term:`glossary link<tau_e_scaling>`

    Returns:
        :term:`energy_confinement_time` [s], :term:`P_in` if tau_e_scaling_uses_P_in=False, else :term:`P_SOL` [MW]
    """

    triangularity_array = np.array([triangularity_psi95, separatrix_triangularity])
    gamma = (
        confinement_time_scalar
        * scaling["C"]
        * plasma_current ** scaling["a_I"]
        * magnetic_field_on_axis ** scaling["a_B"]
        * average_electron_density ** scaling["a_n"]
        * major_radius ** scaling["a_R"]
        * areal_elongation ** scaling["a_ka"]
        * separatrix_elongation ** scaling["a_ks"]
        * inverse_aspect_ratio ** scaling["a_e"]
        * fuel_average_mass_number ** scaling["a_A"]
        * (1.0 + np.mean(triangularity_array)) ** scaling["a_d"]
        * q_star ** scaling["a_q"]
    )

    P_tau = jax.lax.select(
        gamma > 0.0,
        (plasma_stored_energy / gamma) ** (1.0 / (1.0 + scaling["a_P"])),
        np.inf,
    )

    tau_E = plasma_stored_energy / P_tau

    return tau_E, P_tau
