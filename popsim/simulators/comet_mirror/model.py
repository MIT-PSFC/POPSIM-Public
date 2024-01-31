from collections.abc import Sequence
from typing import Callable

import equinox as eqx
import jax

from cfspopcon.jax_compatible import average_fuel_ion_mass, beta, current_drive, fusion_rates, geometry, radiated_power
from cfspopcon.jax_compatible.energy_confinement_time_scalings import tau_e_from_Wp
from cfspopcon.named_options import ConfinementScaling
from popsim.algorithms.density import DensityModelImpurityCalc, MultiSpeciesDensityModel
from popsim.algorithms.profiles import ProfileCalculator
from popsim.enums import FuelSpecies, Impurity, ProfileForm, ReactionType, Species


class State(eqx.Module):
    stored_energy: float  # [MJ]
    density_state: MultiSpeciesDensityModel.State


class Params(eqx.Module):
    density_params: MultiSpeciesDensityModel.Params
    major_radius: float  # [m]
    magnetic_field_on_axis: float  # [T]
    inverse_aspect_ratio: float  # [-]
    areal_elongation: float  # [-]
    elongation_ratio_sep_to_areal: float  # [-]
    triangularity_psi95: float  # [-]
    triangularity_ratio_sep_to_psi95: float  # [-]
    plasma_current: float  # [A]
    fraction_of_external_power_coupled: float  # [-]
    normalized_inverse_temp_scale_length: float  # [-]
    electron_density_peaking_offset: float  # [-]
    ion_density_peaking_offset: float  # [-]
    temperature_peaking: float  # [-]
    ion_to_electron_temp_ratio: float  # [-]
    confinement_time_scalar: float  # [-]
    P_aux_MW: float  # [MW]


class CometMirror(eqx.Module):
    density_model: MultiSpeciesDensityModel
    impurity_calc: DensityModelImpurityCalc
    profile_form: ProfileForm
    profiles: ProfileCalculator
    fusion_reaction: ReactionType
    calc_tau_e_and_P_in_from_scaling: Callable
    calc_fuel_average_mass_number: Callable

    def __init__(
        self,
        species: Sequence[Species],
        profile_form: ProfileForm,
        energy_confinement_scaling: ConfinementScaling,
        fusion_reaction: ReactionType = ReactionType.DT,
    ):
        self.density_model = MultiSpeciesDensityModel(species)
        self.impurity_calc = DensityModelImpurityCalc(species)
        self.profile_form = profile_form
        self.profiles = ProfileCalculator(profile_form=profile_form, n_points=100)
        self.fusion_reaction = fusion_reaction
        self.calc_tau_e_and_P_in_from_scaling = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(scaling=energy_confinement_scaling)

        def calc_fuel_average_mass_number(
            heavier_fuel_species_fraction: float,
        ) -> float:
            jitted_fn = jax.jit(average_fuel_ion_mass.calc_fuel_average_mass_number, static_argnames=["fusion_reaction"])
            return jitted_fn(fusion_reaction=fusion_reaction, heavier_fuel_species_fraction=heavier_fuel_species_fraction)

        self.calc_fuel_average_mass_number = calc_fuel_average_mass_number

    def __call__(self, state: State, params: Params) -> State:
        """Geometric calculations."""
        plasma_volume = geometry.calc_plasma_volume(
            major_radius=params.major_radius,
            inverse_aspect_ratio=params.inverse_aspect_ratio,
            areal_elongation=params.areal_elongation,
        )
        separatrix_triangularity = params.triangularity_psi95 * params.triangularity_ratio_sep_to_psi95
        minor_radius = params.major_radius * params.inverse_aspect_ratio
        separatrix_elongation = params.areal_elongation * params.elongation_ratio_sep_to_areal

        """Calculate q_star."""
        f_shaping = current_drive.calc_f_shaping(params.inverse_aspect_ratio, params.areal_elongation, params.triangularity_psi95)
        q_star = current_drive.calc_q_star(
            params.magnetic_field_on_axis,
            params.major_radius,
            params.inverse_aspect_ratio,
            # Convert to MA.
            1e-6 * params.plasma_current,
            f_shaping,
        )

        """Calculate kinetics."""
        average_stored_energy_Joule = 1e6 * state.stored_energy / plasma_volume
        EV_TO_JOULE = 1.6022e-19
        average_stored_energy_eV = average_stored_energy_Joule / EV_TO_JOULE
        average_stored_energy_keV = average_stored_energy_eV / 1e3
        average_stored_energy_keV_1e19 = average_stored_energy_keV / 1e19

        # Friedberg 14.150.
        average_pressure_keV_1e19 = (2.0 / 3.0) * average_stored_energy_keV_1e19

        # Calculate impurity related quantities.
        impurity_out = self.impurity_calc(density_state=state.density_state, average_pressure_kev_1e19=average_pressure_keV_1e19)
        z_effective, dilution, average_electron_density_19 = (
            impurity_out["z_effective"],
            impurity_out["dilution"],
            impurity_out["volume_average_electron_density"],
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
            minor_radius=minor_radius,
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
            major_radius=params.major_radius,
            z_effective=z_effective,
            dilution=dilution,
            beta_toroidal=beta_t,
            normalized_inverse_temp_scale_length=params.normalized_inverse_temp_scale_length,
        )

        """
        Calculate radiation.
        """
        P_rad_bremsstrahlung_MW = radiated_power.calc_bremsstrahlung_radiation(
            rho=self.profiles.rho,
            electron_density_profile=profiles["electron_density_profile"],
            electron_temp_profile=profiles["electron_temp_profile"],
            z_effective=z_effective,
            plasma_volume=plasma_volume,
        )
        P_rad_synchrotron_MW = radiated_power.calc_synchrotron_radiation(
            rho=self.profiles.rho,
            electron_density_profile=profiles["electron_density_profile"],
            electron_temp_profile=profiles["electron_temp_profile"],
            major_radius=params.major_radius,
            minor_radius=minor_radius,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
            separatrix_elongation=separatrix_elongation,
            plasma_volume=plasma_volume,
        )

        Prad_imp_MW, _ = self.impurity_calc.calc_impurity_radiated_power_radas(
            rho=self.profiles.rho,
            # The radas calculation uses eV and m^-3.
            electron_temp_profile=1e3 * profiles["electron_temp_profile"],
            electron_density_profile=1e19 * profiles["electron_density_profile"],
            impurity_concentrations=params.impurity_concentrations,
            plasma_volume=plasma_volume,
        )

        """
        Calculate fusion power.
        """
        # TODO(allenw): make generic for all fuel types.
        heavier_fuel_species_fraction = state.density_state.volume_average_ion_densities[FuelSpecies.Tritium] / (
            state.density_state.volume_average_ion_densities[FuelSpecies.Deuterium]
            + state.density_state.volume_average_ion_densities[FuelSpecies.Tritium]
        )
        P_fusion_MW, P_neutron_MW, P_alpha_MW, reactions_per_second = fusion_rates.calc_fusion_power(
            fusion_reaction=self.fusion_reaction,
            ion_temp_profile=profiles["ion_temp_profile"],
            heavier_fuel_species_fraction=heavier_fuel_species_fraction,
            nfuel19=profiles["ion_density_profile"],
            rho=self.profiles.rho,
            plasma_volume=plasma_volume,
        )

        """Calculate conduction losses."""
        fuel_average_mass_number = self.calc_fuel_average_mass_number(params.heavier_fuel_species_fraction)
        tau_E, P_tau_MW = self.calc_tau_e_and_P_in_from_scaling(
            confinement_time_scalar=params.confinement_time_scalar,
            # Convert to MA.
            plasma_current=1e-6 * params.plasma_current,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
            average_electron_density=average_electron_density_19,
            major_radius=params.major_radius,
            areal_elongation=params.areal_elongation,
            separatrix_elongation=separatrix_elongation,
            inverse_aspect_ratio=params.inverse_aspect_ratio,
            fuel_average_mass_number=fuel_average_mass_number,
            triangularity_psi95=params.triangularity_psi95,
            separatrix_triangularity=separatrix_triangularity,
            # Convert to MJ.
            plasma_stored_energy=state.stored_energy,
            q_star=q_star,
        )

        """Calculate ohmic power."""
        bootstrap_fraction = current_drive.calc_bootstrap_fraction(
            ion_density_peaking=profiles["ion_density_peaking"],
            electron_density_peaking=profiles["electron_density_peaking"],
            temperature_peaking=params.temperature_peaking,
            z_effective=z_effective,
            q_star=q_star,
            inverse_aspect_ratio=params.inverse_aspect_ratio,
            beta_poloidal=beta_p,
        )

        inductive_plasma_current = params.plasma_current * (1.0 - bootstrap_fraction)
        spitzer_resistivity = current_drive.calc_Spitzer_loop_resistivity(average_electron_temp_keV)
        trapped_particle_fraction = current_drive.calc_resistivity_trapped_enhancement(params.inverse_aspect_ratio)
        neoclassical_loop_resistivity = current_drive.calc_neoclassical_loop_resistivity(
            spitzer_resistivity, z_effective, trapped_particle_fraction
        )
        loop_voltage = current_drive.calc_loop_voltage(
            params.major_radius, minor_radius, inductive_plasma_current, params.areal_elongation, neoclassical_loop_resistivity
        )
        P_ohmic_MW = current_drive.calc_ohmic_power(1e-6 * inductive_plasma_current, loop_voltage)

        P_rad_MW = P_rad_bremsstrahlung_MW + P_rad_synchrotron_MW + Prad_imp_MW

        Paux_MW = params.fraction_of_external_power_coupled * params.P_aux_MW

        dW_dt = -P_tau_MW + P_alpha_MW + P_ohmic_MW + Paux_MW - P_rad_MW

        params.density_params.sources_and_sinks[FuelSpecies.Deuterium]["fusion"] = -reactions_per_second
        params.density_params.sources_and_sinks[FuelSpecies.Tritium]["fusion"] = -reactions_per_second
        params.density_params.sources_and_sinks[Impurity.Helium]["fusion"] = reactions_per_second

        density_dot = self.density_model(
            state.density_state,
            params.density_params,
        )

        state_dot = State(
            stored_energy=dW_dt,
            density_state=density_dot,
        )
        return state_dot
