import dataclasses
from typing import Callable

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from cfspopcon.named_options import ConfinementScaling
from jaxtyping import Array

import popsim.modules.density as density_model
import popsim.modules.hmode_dynamics as hmode
from popsim import ModuleBase
from popsim.cfspopcon_jax import average_fuel_ion_mass, beta, current_drive, fusion_rates, radiated_power
from popsim.cfspopcon_jax.confinement_regime_threshold_powers import calc_LH_transition_threshold_power
from popsim.cfspopcon_jax.energy_confinement_time_scalings import tau_e_from_Wp
from popsim.cfspopcon_jax.fusion_rates import ReactionType
from popsim.cfspopcon_jax.helpers import integrate_profile_over_volume_cylindrical
from popsim.enums import FuelSpecies, Impurity, ProfileForm, Species, SpeciesContainer
from popsim.interfaces.atomic_data import RadasCurves, read_atomic_data
from popsim.physics.geometry import GeometryCFSPopcon
from popsim.physics.impurities import calc_impurity_radiated_power_radas, calc_impurity_state
from popsim.physics.profiles import ProfileCalculator


@chex.dataclass
class CometMirror(ModuleBase):
    @chex.dataclass
    class State:
        """
        State variables for the CometMirror model.
        """

        stored_energy: float  # [MJ]
        density_state: density_model.Density.State
        hmode_state: hmode.HmodeDynamics.State

    @chex.dataclass
    class Params:
        """
        Dynamic parameters for the CometMirror model.
        """

        magnetic_field_on_axis: float  # [T]
        plasma_current: float  # [A]
        fraction_of_external_power_coupled: float  # [-]
        normalized_inverse_temp_scale_length: float  # [-]
        electron_density_peaking_offset: float  # [-]
        ion_density_peaking_offset: float  # [-]
        temperature_peaking: float  # [-]
        ion_to_electron_temp_ratio: float  # [-]
        confinement_time_scalar: float  # [-]
        P_aux_MW: float  # [MW]
        geometry: GeometryCFSPopcon
        fueling19: dict[Species, float]  # 1e19/s
        particle_confinement_scalar: dict[Species, float]  # [-]
        hmode_transition_characteristic_time: float  # [s]
        hl_threshold_scalar: float  # Assume the h->l transition is some fraction of the l->h transition [-]

    @chex.dataclass
    class Config:
        """
        Static compile-time configuration for the CometMirror model.
        """

        species: SpeciesContainer
        profile_form: ProfileForm
        rho: Array
        hmode_scaling: ConfinementScaling = ConfinementScaling.ITER98y2
        lmode_scaling: ConfinementScaling = ConfinementScaling.ITER89P_ka
        fusion_reaction: ReactionType = ReactionType.DT
        radas_curves: RadasCurves = dataclasses.field(default_factory=read_atomic_data)

    @chex.dataclass
    class Output:
        aux_data: dict[str, Array]

    config: Config
    profiles: ProfileCalculator
    hmode_tau_e_and_P: Callable
    lmode_tau_e_and_P: Callable
    calc_fuel_average_mass_number: Callable

    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.profiles = ProfileCalculator(profile_form=self.config.profile_form, rho=self.config.rho)
        self.hmode_tau_e_and_P = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(scaling=self.config.hmode_scaling)
        self.lmode_tau_e_and_P = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(scaling=self.config.lmode_scaling)

        def calc_fuel_average_mass_number(
            heavier_fuel_species_fraction: float,
        ) -> float:
            jitted_fn = jax.jit(average_fuel_ion_mass.calc_fuel_average_mass_number, static_argnames=["fusion_reaction"])
            return jitted_fn(fusion_reaction=self.config.fusion_reaction, heavier_fuel_species_fraction=heavier_fuel_species_fraction)

        self.calc_fuel_average_mass_number = calc_fuel_average_mass_number

    def __call__(self, state: State, params: Params) -> State:
        """Calculate q_star."""
        q_star = current_drive.calc_q_star(
            params.magnetic_field_on_axis,
            params.geometry.major_radius,
            params.geometry.inverse_aspect_ratio,
            # Convert to MA.
            1e-6 * params.plasma_current,
            params.geometry.f_shaping,
        )

        """Calculate kinetics."""
        average_stored_energy_Joule = 1e6 * state.stored_energy / params.geometry.plasma_volume
        EV_TO_JOULE = 1.6022e-19
        average_stored_energy_eV = average_stored_energy_Joule / EV_TO_JOULE
        average_stored_energy_keV = average_stored_energy_eV / 1e3
        average_stored_energy_keV_1e19 = average_stored_energy_keV / 1e19

        # Friedberg 14.150.
        average_pressure_keV_1e19 = (2.0 / 3.0) * average_stored_energy_keV_1e19

        # Calculate impurity related quantities.
        impurity_out = calc_impurity_state(
            density_state=state.density_state, average_pressure_kev_1e19=average_pressure_keV_1e19, radas_curves=self.config.radas_curves
        )
        z_effective, dilution, average_electron_density_19, species_concentrations = (
            impurity_out["z_effective"],
            impurity_out["dilution"],
            impurity_out["volume_average_electron_density_19"],
            impurity_out["species_concentrations"],
        )

        average_electron_temp_keV = average_pressure_keV_1e19 / (
            average_electron_density_19 + state.density_state.total_volume_average_ion_density * params.ion_to_electron_temp_ratio
        )
        average_ion_temp_keV = params.ion_to_electron_temp_ratio * average_electron_temp_keV

        beta_p = beta.calc_beta_poloidal(
            average_electron_density=average_electron_density_19,
            average_electron_temp=average_electron_temp_keV,
            average_ion_temp=average_ion_temp_keV,
            # Convert to MA.
            plasma_current=1e-6 * params.plasma_current,
            minor_radius=params.geometry.minor_radius,
        )

        beta_t = beta.calc_beta_toroidal(
            average_electron_density=average_electron_density_19,
            average_electron_temp=average_electron_temp_keV,
            average_ion_temp=average_ion_temp_keV,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
        )

        profiles = self.profiles(
            average_electron_density_19=average_electron_density_19,
            average_electron_temp_keV=average_electron_temp_keV,
            average_ion_temp_keV=average_ion_temp_keV,
            ion_density_peaking_offset=params.ion_density_peaking_offset,
            electron_density_peaking_offset=params.electron_density_peaking_offset,
            temperature_peaking=params.temperature_peaking,
            major_radius=params.geometry.major_radius,
            z_effective=z_effective,
            dilution=dilution,
            beta_toroidal=beta_t,
            normalized_inverse_temp_scale_length=params.normalized_inverse_temp_scale_length,
        )

        """
        Calculate radiation.
        """

        def volume_integrator(quantity_per_m3) -> float:
            # TODO(allenw): currently using cylindrical. Eventually incorporate dV/drho.
            return integrate_profile_over_volume_cylindrical(
                quantity_per_m3, rho=profiles["rho"].data, plasma_volume=params.geometry.plasma_volume
            )

        P_rad_bremsstrahlung_MW = radiated_power.calc_bremsstrahlung_radiation(
            electron_density_profile=profiles["electron_density_profile"].data,
            electron_temp_profile=profiles["electron_temp_profile"].data,
            z_effective=z_effective,
            volume_integrator=volume_integrator,
        )
        P_rad_synchrotron_MW = radiated_power.calc_synchrotron_radiation(
            electron_density_profile=profiles["electron_density_profile"].data,
            electron_temp_profile=profiles["electron_temp_profile"].data,
            major_radius=params.geometry.major_radius,
            minor_radius=params.geometry.minor_radius,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
            separatrix_elongation=params.geometry.separatrix_elongation,
            volume_integrator=volume_integrator,
        )

        Prad_imp_MW, imp_debugs = calc_impurity_radiated_power_radas(
            # The radas calculation uses eV and m^-3.
            electron_temp_profile=1e3 * profiles["electron_temp_profile"].data,
            electron_density_profile=1e19 * profiles["electron_density_profile"].data,
            impurity_concentrations={k: v for k, v in species_concentrations.items() if isinstance(k, Impurity)},
            volume_integrator=volume_integrator,
            radas_curves=self.config.radas_curves,
        )

        """
        Calculate fusion power.
        """
        # TODO(allenw): make generic for all fuel types.
        # Although... if advanced fuel reactions become relevant, that will be a great problem to have :).
        heavier_fuel_species_fraction = state.density_state.vol_avg_ion[FuelSpecies.Tritium] / (
            state.density_state.vol_avg_ion[FuelSpecies.Deuterium] + state.density_state.vol_avg_ion[FuelSpecies.Tritium]
        )
        P_fusion_MW, P_neutron_MW, P_alpha_MW, reactions_per_second = fusion_rates.calc_fusion_power(
            fusion_reaction=self.config.fusion_reaction,
            ion_temp_profile=profiles["ion_temp_profile"].data,
            heavier_fuel_species_fraction=heavier_fuel_species_fraction,
            nfuel19=profiles["ion_density_profile"].data,
            volume_integrator=volume_integrator,
        )

        # Convert reactions_per_second to 1e19/s.
        reactions_per_second = reactions_per_second / 1e19

        """Calculate tauE and conduction losses"""
        fuel_average_mass_number = self.calc_fuel_average_mass_number(heavier_fuel_species_fraction)

        def calc_with_scaling_law_fun(scaling_law_fun):
            tau_E, P_tau_MW = scaling_law_fun(
                confinement_time_scalar=params.confinement_time_scalar,
                # Convert to MA.
                plasma_current=1e-6 * params.plasma_current,
                magnetic_field_on_axis=params.magnetic_field_on_axis,
                average_electron_density=average_electron_density_19,
                major_radius=params.geometry.major_radius,
                areal_elongation=params.geometry.areal_elongation,
                separatrix_elongation=params.geometry.separatrix_elongation,
                inverse_aspect_ratio=params.geometry.inverse_aspect_ratio,
                fuel_average_mass_number=fuel_average_mass_number,
                triangularity_psi95=params.geometry.triangularity_psi95,
                separatrix_triangularity=params.geometry.separatrix_triangularity,
                # Convert to MJ.
                plasma_stored_energy=state.stored_energy,
                q_star=q_star,
            )
            return jnp.array([tau_E, P_tau_MW])

        in_hmode = state.hmode_state.in_hmode
        tau_E, P_tau_MW = jnp.where(
            in_hmode,
            calc_with_scaling_law_fun(self.hmode_tau_e_and_P),
            calc_with_scaling_law_fun(self.lmode_tau_e_and_P),
        )

        """Calculate ohmic power."""
        bootstrap_fraction = current_drive.calc_bootstrap_fraction(
            ion_density_peaking=profiles["ion_density_peaking"],
            electron_density_peaking=profiles["electron_density_peaking"],
            temperature_peaking=params.temperature_peaking,
            z_effective=z_effective,
            q_star=q_star,
            inverse_aspect_ratio=params.geometry.inverse_aspect_ratio,
            beta_poloidal=beta_p,
        )

        inductive_plasma_current = params.plasma_current * (1.0 - bootstrap_fraction)
        spitzer_resistivity = current_drive.calc_Spitzer_loop_resistivity(average_electron_temp_keV)
        trapped_particle_fraction = current_drive.calc_resistivity_trapped_enhancement(params.geometry.inverse_aspect_ratio)
        neoclassical_loop_resistivity = current_drive.calc_neoclassical_loop_resistivity(
            spitzer_resistivity, z_effective, trapped_particle_fraction
        )
        loop_voltage = current_drive.calc_loop_voltage(
            params.geometry.major_radius,
            params.geometry.minor_radius,
            inductive_plasma_current,
            params.geometry.areal_elongation,
            neoclassical_loop_resistivity,
        )
        P_ohmic_MW = current_drive.calc_ohmic_power(1e-6 * inductive_plasma_current, loop_voltage)

        P_rad_MW = P_rad_bremsstrahlung_MW + P_rad_synchrotron_MW + Prad_imp_MW

        Paux_MW = params.fraction_of_external_power_coupled * params.P_aux_MW

        dW_dt = -P_tau_MW + P_alpha_MW + P_ohmic_MW + Paux_MW - P_rad_MW

        sources_and_sinks = {k: {} for k in self.config.species.species}
        for k, v in params.fueling19.items():
            sources_and_sinks[k]["fueling19"] = v

        sources_and_sinks[FuelSpecies.Deuterium]["fusion"] = -reactions_per_second
        sources_and_sinks[FuelSpecies.Tritium]["fusion"] = -reactions_per_second
        sources_and_sinks[Impurity.Helium]["fusion"] = reactions_per_second

        density_params = density_model.Density.Params(
            sources_and_sinks=sources_and_sinks,
            species_confinement_time=jax.tree.map(lambda k: k * tau_E, params.particle_confinement_scalar),
            volume_dot=0.0,  # TODO(allenw): add with time-varying geometry.
            volume=params.geometry.plasma_volume,
        )

        density_dot = density_model.multi_species_derivs(state.density_state, density_params)

        """Calculate H + L mode dynamics."""
        lh_threshold = calc_LH_transition_threshold_power(
            plasma_current=1e-6 * params.plasma_current,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
            minor_radius=params.geometry.minor_radius,
            major_radius=params.geometry.major_radius,
            surface_area=params.geometry.surface_area,
            fuel_average_mass_number=fuel_average_mass_number,
            average_electron_density=average_electron_density_19,
        )
        hmode_params = hmode.HmodeDynamics.Params(
            transition_characteristic_time=params.hmode_transition_characteristic_time,
            P_tau_MW=jnp.abs(P_tau_MW),
            P_input_MW=jnp.abs(Paux_MW + P_ohmic_MW + P_alpha_MW),
            lh_threshold_MW=lh_threshold,
            hl_threshold_MW=params.hl_threshold_scalar
            * lh_threshold,  # Assume the h->l transition is some fraction of the l->h transition.
        )

        hmode_dot = hmode.dynamics(state.hmode_state, hmode_params)

        state_dot = CometMirror.State(
            stored_energy=dW_dt,
            density_state=density_dot,
            hmode_state=hmode_dot,
        )

        # Filter out any non-array-like variables
        aux_data = eqx.filter(locals(), eqx.is_array_like)
        # Promote any scalar-like variables to arrays
        aux_data = jax.tree.map(jnp.asarray, aux_data)
        output = CometMirror.Output(aux_data=aux_data)
        return state_dot, output
