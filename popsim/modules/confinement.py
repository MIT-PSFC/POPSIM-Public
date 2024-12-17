from typing import Callable

import chex
import jax
import jax.numpy as jnp
from cfspopcon.formulas.energy_confinement.read_energy_confinement_scalings import ConfinementScaling, read_confinement_scalings

from popsim.cfspopcon_jax.confinement_regime_threshold_powers import calc_LH_transition_threshold_power
from popsim.cfspopcon_jax.energy_confinement_time_scalings import tau_e_from_Wp
from popsim.enums import Species
from popsim.modules.hmode_dynamics import HmodeDynamics
from popsim.physics.geometry import GeometryCFSPopcon

CRITICAL_THRESHOLD = 0.5


@chex.dataclass
class Confinement:
    @chex.dataclass
    class Params:
        magnetic_field_on_axis: float  # [T]
        plasma_current: float  # [A]
        stored_energy: float  # [MJ]
        average_electron_density_19: float  # [1e19/m^3]
        confinement_time_scalar: float  # [-]
        geometry: GeometryCFSPopcon
        particle_confinement_scalar: dict[Species, float]  # [-]
        hmode_transition_characteristic_time: float  # [s]
        hl_threshold_scalar: float  # Assume the h->l transition is some fraction of the l->h transition [-]
        heavier_fuel_species_fraction: float  # [-]
        fuel_average_mass_number: float  # [-]
        q_star: float  # [-]
        transition_characteristic_time: chex.Numeric  # amount of time for a L->H or H->L transition to occur [s].
        P_tau_MW: chex.Numeric  # Power conducted to the scrape-off layer [MW]
        P_input_MW: chex.Numeric  # Power input to the plasma [MW]

    @chex.dataclass
    class State:
        hmode_state: HmodeDynamics.State

        @property
        def in_hmode(self):
            return jnp.bool_(self.hmode_state.hmode >= CRITICAL_THRESHOLD)

    @chex.dataclass
    class Output:
        tau_E: float  # [s]
        in_hmode: bool  # [-]
        species_confinement_time: dict[Species, float]  # species confinement time [s]

    @chex.dataclass
    class Config:
        hmode_scaling: ConfinementScaling = ConfinementScaling.instances["ITER98y2"]
        lmode_scaling: ConfinementScaling = ConfinementScaling.instances["ITER89P_ka"]

    read_confinement_scalings()
    config: Config
    hmode_tau_e_and_P: Callable
    lmode_tau_e_and_P: Callable
    hmode_dynamics: HmodeDynamics

    def __init__(self, config):
        self.config = config
        self.hmode_tau_e_and_P = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(scaling=self.config.hmode_scaling)
        self.lmode_tau_e_and_P = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(scaling=self.config.lmode_scaling)
        self.hmode_dynamics = HmodeDynamics(config=HmodeDynamics.Config())

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        lh_threshold_MW = calc_LH_transition_threshold_power(
            plasma_current=1e-6 * params.plasma_current,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
            minor_radius=params.geometry.minor_radius,
            major_radius=params.geometry.major_radius,
            surface_area=params.geometry.surface_area,
            fuel_average_mass_number=params.fuel_average_mass_number,
            average_electron_density=params.average_electron_density_19,
        )

        hmode_params = HmodeDynamics.Params(
            transition_characteristic_time=params.hmode_transition_characteristic_time,
            P_tau_MW=jnp.abs(params.P_tau_MW),
            P_input_MW=jnp.abs(params.P_input_MW),
            lh_threshold_MW=lh_threshold_MW,
            hl_threshold_MW=params.hl_threshold_scalar
            * lh_threshold_MW,  # Assume the h->l transition is some fraction of the l->h transition.
        )

        hmode_dot, hmode_out = self.hmode_dynamics(state.hmode_state, hmode_params)

        in_hmode = state.hmode_state.in_hmode
        tau_E, P_tau_MW = jnp.where(
            in_hmode,
            calc_with_scaling_law_fun(self.hmode_tau_e_and_P, state, params),
            calc_with_scaling_law_fun(self.lmode_tau_e_and_P, state, params),
        )
        species_confinement_time = jax.tree.map(lambda k: k * tau_E, params.particle_confinement_scalar)

        return Confinement.State(hmode_state=hmode_dot), Confinement.Output(
            tau_E=tau_E, species_confinement_time=species_confinement_time, in_hmode=in_hmode
        )


"""Calculate tauE and conduction losses"""


def calc_with_scaling_law_fun(
    scaling_law_fun, state: Confinement.State, params: Confinement.Params
):  # TODO: Just use scaling directly (w/ power as input)
    tau_E, P_tau_MW = scaling_law_fun(
        confinement_time_scalar=params.confinement_time_scalar,
        # Convert to MA.
        plasma_current=1e-6 * params.plasma_current,
        magnetic_field_on_axis=params.magnetic_field_on_axis,
        average_electron_density=params.average_electron_density_19,
        major_radius=params.geometry.major_radius,
        areal_elongation=params.geometry.areal_elongation,
        separatrix_elongation=params.geometry.separatrix_elongation,
        inverse_aspect_ratio=params.geometry.inverse_aspect_ratio,
        fuel_average_mass_number=params.fuel_average_mass_number,
        triangularity_psi95=params.geometry.triangularity_psi95,
        separatrix_triangularity=params.geometry.separatrix_triangularity,
        # Convert to MJ.
        plasma_stored_energy=params.stored_energy,
        q_star=params.q_star,
    )
    return jnp.array([tau_E, P_tau_MW])
