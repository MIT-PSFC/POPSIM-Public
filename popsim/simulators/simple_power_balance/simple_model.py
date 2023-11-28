from collections.abc import Sequence
from typing import Callable

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from cfspopcon.jax_compatible import average_fuel_ion_mass, beta, current_drive, fusion_rates, geometry, radiated_power
from cfspopcon.jax_compatible.energy_confinement_time_scalings import tau_e_from_Wp
from cfspopcon.jax_compatible.fusion_rates import ReactionType
from cfspopcon.named_options import ConfinementScaling
from popsim.algorithms.profiles import ProfileCalculator
from popsim.algorithms.zeff_and_dilution_from_impurities import (
    CalcZeffAndDilutionFromImpurities,
    TempDensImpurities,
)
from popsim.enums import Impurity


class State(eqx.Module):
    stored_energy: float  # [MJ]


class Params(eqx.Module):
    major_radius: float  # [m]
    magnetic_field_on_axis: float  # [T]
    inverse_aspect_ratio: float  # [-]
    areal_elongation: float  # [-]
    elongation_ratio_sep_to_areal: float  # [-]
    triangularity_psi95: float  # [-]
    triangularity_ratio_sep_to_psi95: float  # [-]
    plasma_current: float  # [A]
    fraction_of_external_power_coupled: float  # [-]
    heavier_fuel_species_fraction: float  # [-]
    normalized_inverse_temp_scale_length: float  # [-]
    electron_density_peaking_offset: float  # [-]
    ion_density_peaking_offset: float  # [-]
    temperature_peaking: float  # [-]
    ion_to_electron_temp_ratio: float  # [-]
    impurity_concentrations: dict[Impurity, float]  # [-]
    confinement_time_scalar: float  # [-]
    P_aux_MW: float  # [MW]
    average_ion_density: float  # [1e19 m^-3]


class SimpleModel(eqx.Module):
    fusion_reaction: ReactionType
    impurity_calc: CalcZeffAndDilutionFromImpurities
    temp_dens_imp: TempDensImpurities
    profiles: ProfileCalculator
    calc_tau_e_and_P_in_from_scaling: Callable
    calc_fuel_average_mass_number: Callable

    def __init__(
        self,
        impurities: Sequence[Impurity],
        energy_confinement_scaling: ConfinementScaling,
        fusion_reaction: ReactionType = ReactionType.DT,
    ):
        self.fusion_reaction = fusion_reaction
        self.impurity_calc = CalcZeffAndDilutionFromImpurities(impurities)
        self.temp_dens_imp = TempDensImpurities(
            zeff_and_dilution_calc=self.impurity_calc,
        )

        self.profiles = ProfileCalculator(n_points=100)

        self.calc_tau_e_and_P_in_from_scaling = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(scaling=energy_confinement_scaling)

        def calc_fuel_average_mass_number(
            heavier_fuel_species_fraction: float,
        ) -> float:
            jitted_fn = jax.jit(average_fuel_ion_mass.calc_fuel_average_mass_number, static_argnames=["fusion_reaction"])
            return jitted_fn(fusion_reaction=fusion_reaction, heavier_fuel_species_fraction=heavier_fuel_species_fraction)

        self.calc_fuel_average_mass_number = calc_fuel_average_mass_number

    def __call__(self, state: State, params: Params, debug_info: bool = False) -> State:  # noqa: PLR0915
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

        outs = self.temp_dens_imp(
            average_pressure=average_pressure_keV_1e19,
            average_ion_density=params.average_ion_density,
            ion_to_electron_temp_ratio=params.ion_to_electron_temp_ratio,
            impurity_concentrations=params.impurity_concentrations,
        )
        average_electron_density = outs["average_electron_density"]
        average_electron_temp = outs["average_electron_temp"]
        average_ion_density = outs["average_ion_density"]
        average_ion_temp = outs["average_ion_temp"]
        z_effective = outs["z_effective"]
        dilution = outs["dilution"]
        summed_impurity_density = outs["summed_impurity_density"]  # noqa: F841

        beta_p = beta.calc_beta_poloidal(
            average_electron_density=average_electron_density,
            average_electron_temp=average_electron_temp,
            average_ion_temp=average_ion_temp,
            # Convert to MA.
            plasma_current=1e-6 * params.plasma_current,
            minor_radius=minor_radius,
        )

        beta_t = beta.calc_beta_toroidal(
            average_electron_density=average_electron_density,
            average_electron_temp=average_electron_temp,
            average_ion_temp=average_ion_temp,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
        )

        profiles = self.profiles(
            average_electron_density_19=average_electron_density,
            average_electron_temp_keV=average_electron_temp,
            average_ion_temp_keV=average_ion_temp,
            ion_density_peaking_offset=params.ion_density_peaking_offset,
            electron_density_peaking_offset=params.electron_density_peaking_offset,
            temperature_peaking=params.temperature_peaking,
            major_radius=params.major_radius,
            z_effective=z_effective,
            dilution=dilution,
            beta_toroidal=beta_t,
        )
        electron_density_profile = profiles["electron_density_profile"]
        ion_density_profile = profiles["ion_density_profile"]
        electron_temp_profile = profiles["electron_temp_profile"]
        ion_temp_profile = profiles["ion_temp_profile"]

        """
        Calculate radiation.
        """
        P_rad_bremsstrahlung_MW = radiated_power.calc_bremsstrahlung_radiation(
            rho=self.profiles.rho,
            electron_density_profile=electron_density_profile,
            electron_temp_profile=electron_temp_profile,
            z_effective=z_effective,
            plasma_volume=plasma_volume,
        )
        P_rad_synchrotron_MW = radiated_power.calc_synchrotron_radiation(
            rho=self.profiles.rho,
            electron_density_profile=electron_density_profile,
            electron_temp_profile=electron_temp_profile,
            major_radius=params.major_radius,
            minor_radius=minor_radius,
            magnetic_field_on_axis=params.magnetic_field_on_axis,
            separatrix_elongation=separatrix_elongation,
            plasma_volume=plasma_volume,
        )

        Prad_imp_MW, _ = self.impurity_calc.calc_impurity_radiated_power_radas(
            rho=self.profiles.rho,
            # The radas calculation uses eV and m^-3.
            electron_temp_profile=1e3 * electron_temp_profile,
            electron_density_profile=1e19 * electron_density_profile,
            impurity_concentrations=params.impurity_concentrations,
            plasma_volume=plasma_volume,
        )

        """
        Calculate fusion power.
        """
        P_fusion_MW, P_neutron_MW, P_alpha_MW = fusion_rates.calc_fusion_power(
            fusion_reaction=self.fusion_reaction,
            ion_temp_profile=ion_temp_profile,
            heavier_fuel_species_fraction=params.heavier_fuel_species_fraction,
            nfuel19=ion_density_profile,
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
            average_electron_density=average_electron_density,
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
        spitzer_resistivity = current_drive.calc_Spitzer_loop_resistivity(average_electron_temp)
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

        derivs = State(
            stored_energy=dW_dt,
        )

        if not debug_info:
            return derivs
        else:
            debug = {
                "P_rad_bremsstrahlung_MW": P_rad_bremsstrahlung_MW,
                "P_rad_synchrotron_MW": P_rad_synchrotron_MW,
                "Prad_imp_MW": Prad_imp_MW,
                "P_fusion_MW": P_fusion_MW,
                "P_neutron_MW": P_neutron_MW,
                "P_alpha_MW": P_alpha_MW,
                "P_ohmic_MW": P_ohmic_MW,
                "P_tau_MW": P_tau_MW,
                "tau_E": tau_E,
                "beta_p": beta_p,
                "beta_t": beta_t,
                "average_electron_density": average_electron_density,
                "average_electron_temp": average_electron_temp,
                "average_ion_density": average_ion_density,
                "average_ion_temp": average_ion_temp,
                "z_effective": z_effective,
                "dilution": dilution,
                "profiles": profiles,
                "q_star": q_star,
            }
        return derivs, debug


class Simulator(eqx.Module):
    model: SimpleModel
    term: diffrax.ODETerm

    def __init__(self, model: SimpleModel):
        self.model = model

        def model_f(t, y, args):
            out = model(y, *args)
            return out

        self.term = diffrax.ODETerm(model_f)

    def __call__(self, ts, state0, params):
        sol = diffrax.diffeqsolve(
            terms=self.term,
            solver=diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=jnp.min(jnp.diff(ts)),
            y0=state0,
            args=(params,),
            saveat=diffrax.SaveAt(ts=ts),
        )
        derivs, debugs = jax.vmap(lambda y: self.model(y, params, debug_info=True))(sol.ys)
        return sol, derivs, debugs
