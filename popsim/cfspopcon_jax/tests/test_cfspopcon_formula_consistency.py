"""Basic tests that functions can be compiled with JAX."""

import cfspopcon.formulas as formulas_og
import cfspopcon.named_options as cfsno
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from cfspopcon.formulas.energy_confinement.read_energy_confinement_scalings import ConfinementScaling, read_confinement_scalings
from cfspopcon.unit_handling import Quantity

from popsim.cfspopcon_jax import (
    Q_thermal_gain_factor,
    average_fuel_ion_mass,
    beta,
    confinement_regime_threshold_powers,
    current_drive,
    density_peaking,
    divertor_metrics,
    figures_of_merit,
    fusion_rates,
    geometry,
    helpers,
    impurity_effects,
    operational_limits,
    plasma_profiles,
)
from popsim.cfspopcon_jax.energy_confinement_time_scalings import tau_e_from_Wp
from popsim.cfspopcon_jax.fusion_reaction_data import reaction_energies, reaction_rate_coefficients
from popsim.cfspopcon_jax.helpers import integrate_profile_over_volume_cylindrical
from popsim.cfspopcon_jax.plasma_profile_data import density_and_temperature_profile_fits
from popsim.cfspopcon_jax.radiated_power import inherent, radas
from popsim.cfspopcon_jax.scrape_off_layer_model import (
    lambda_q,
    parallel_heat_flux_density,
    solve_target_first_two_point_model,
    solve_two_point_model,
)

DEFAULT_QUANTITIES = {
    "average_electron_density": Quantity(31, "n19"),
    "average_electron_temp": Quantity(7.3, "keV"),
    "average_ion_temp": Quantity(7.3, "keV"),
    "plasma_current": Quantity(8.7, "MA"),
    "major_radius": Quantity(1.85, "m"),
    "minor_radius": Quantity(0.57, "m"),
    "magnetic_field_on_axis": Quantity(12.2, "T"),
    "z_effective": Quantity(1.5, "dimensionless"),
    "ion_density_peaking_offset": Quantity(-0.2, "dimensionless"),
    "separatrix_elongation": Quantity(1.8, "dimensionless"),
    "surface_area": Quantity(20.0, "m^3"),
    "plasma_volume": Quantity(20.0, "m^3"),
    "inverse_aspect_ratio": Quantity(0.3081, "dimensionless"),
    "areal_elongation": Quantity(1.75, "dimensionless"),
    "triangularity_psi95": Quantity(0.3, "dimensionless"),
    "fuel_average_mass_number": Quantity(2.5, "amu"),
}
DEFAULT_MAGS = {k: v.magnitude for k, v in DEFAULT_QUANTITIES.items()}
DEFAULT_RHO = jnp.linspace(0.0, 1.0, 50)


def _DEFAULT_VOLUME_INTEGRATOR(quantity_per_m3):
    return integrate_profile_over_volume_cylindrical(quantity_per_m3, rho=DEFAULT_RHO, plasma_volume=DEFAULT_MAGS["plasma_volume"])


def jax_compatability_test(fn, static_argnames=None, fn_args=None, fn_kwargs=None):
    """Test that a function can be compiled with JAX."""
    if fn_args is None:
        fn_args = ()
    if fn_kwargs is None:
        fn_kwargs = {}

    res0 = fn(*fn_args, **fn_kwargs)
    res1 = jax.jit(fn, static_argnames=static_argnames)(*fn_args, **fn_kwargs)
    chex.assert_trees_all_close(res0, res1)


def test_fusion_rates():
    ion_temp_profile = jnp.linspace(20.0, 1.0, 50)
    nfuel19_profile = jnp.linspace(40, 10, 50)
    inputs = {
        "fusion_reaction": fusion_rates.ReactionType.DT,
        "ion_temp_profile": ion_temp_profile,
        "heavier_fuel_species_fraction": 0.5,
        "nfuel19": nfuel19_profile,
        "volume_integrator": _DEFAULT_VOLUME_INTEGRATOR,
    }
    jax_compatability_test(fusion_rates.calc_fusion_power, static_argnames=["fusion_reaction", "volume_integrator"], fn_kwargs=inputs)


def test_beta():
    def calc_beta_total_and_n():
        betat = beta.calc_beta_toroidal(
            average_electron_density=DEFAULT_MAGS["average_electron_density"],
            average_electron_temp=DEFAULT_MAGS["average_electron_temp"],
            average_ion_temp=DEFAULT_MAGS["average_ion_temp"],
            magnetic_field_on_axis=DEFAULT_MAGS["magnetic_field_on_axis"],
        )
        betap = beta.calc_beta_poloidal(
            average_electron_density=DEFAULT_MAGS["average_electron_density"],
            average_electron_temp=DEFAULT_MAGS["average_electron_temp"],
            average_ion_temp=DEFAULT_MAGS["average_ion_temp"],
            plasma_current=DEFAULT_MAGS["plasma_current"],
            minor_radius=DEFAULT_MAGS["minor_radius"],
        )
        beta_total = beta.calc_beta_total(betat, betap)
        betan = beta.calc_beta_normalised(
            beta=beta_total,
            minor_radius=DEFAULT_MAGS["minor_radius"],
            magnetic_field_on_axis=DEFAULT_MAGS["magnetic_field_on_axis"],
            plasma_current=DEFAULT_MAGS["plasma_current"],
        )
        return beta_total, betan

    jax_compatability_test(calc_beta_total_and_n)

    """Check against the original cfspopcon."""
    beta_total, betan = calc_beta_total_and_n()

    betatog = formulas_og.plasma_pressure.beta.calc_beta_toroidal(
        average_electron_density=DEFAULT_QUANTITIES["average_electron_density"],
        average_electron_temp=DEFAULT_QUANTITIES["average_electron_temp"],
        average_ion_temp=DEFAULT_QUANTITIES["average_ion_temp"],
        magnetic_field_on_axis=DEFAULT_QUANTITIES["magnetic_field_on_axis"],
    )
    betapog = formulas_og.plasma_pressure.beta.calc_beta_poloidal(
        average_electron_density=DEFAULT_QUANTITIES["average_electron_density"],
        average_electron_temp=DEFAULT_QUANTITIES["average_electron_temp"],
        average_ion_temp=DEFAULT_QUANTITIES["average_ion_temp"],
        plasma_current=DEFAULT_QUANTITIES["plasma_current"],
        minor_radius=DEFAULT_QUANTITIES["minor_radius"],
    )
    beta_total_og = formulas_og.plasma_pressure.beta.calc_beta_total(betatog, betapog)
    betan_og = formulas_og.plasma_pressure.beta.calc_beta_normalized(
        beta_total=beta_total_og,
        minor_radius=DEFAULT_QUANTITIES["minor_radius"],
        magnetic_field_on_axis=DEFAULT_QUANTITIES["magnetic_field_on_axis"],
        plasma_current=DEFAULT_QUANTITIES["plasma_current"],
    )

    jnp.isclose(beta_total, beta_total_og.magnitude)
    jnp.isclose(betan, betan_og.magnitude)


def test_geometry():
    jax_compatability_test(geometry.calc_plasma_volume, fn_args=(1.0, 1.0, 1.0))
    jax_compatability_test(geometry.calc_plasma_surface_area, fn_args=(1.0, 1.0, 1.0))


def test_impurity_effects():
    jax_compatability_test(impurity_effects.calc_change_in_zeff, fn_args=(1.0, 1.0))
    jax_compatability_test(impurity_effects.calc_change_in_dilution, fn_args=(1.0, 1.0))

    class DummyFn(eqx.Module):
        def __init__(self):
            super().__init__()

        def __call__(self, x, y):
            return 1.0

    jax_compatability_test(impurity_effects.calc_impurity_charge_state_impl, fn_args=(1.0, 1.0, DummyFn()))


def test_plasma_profiles():
    kwargs = {
        "average_electron_density": DEFAULT_MAGS["average_electron_density"],
        "average_electron_temp": DEFAULT_MAGS["average_electron_temp"],
        "average_ion_temp": DEFAULT_MAGS["average_ion_temp"],
        "electron_density_peaking": 1.0,
        "ion_density_peaking": 1.0,
        "temperature_peaking": 1.0,
        "dilution": 1.0,
        "rho": jnp.linspace(0.0, 1.0, 50),
    }
    jax_compatability_test(plasma_profiles.calc_analytic_profiles, fn_kwargs=kwargs)


def test_profile_fits():
    width_interpolator, aLT_interpolator = density_and_temperature_profile_fits.read_prf_data("PRF")
    x_a = width_interpolator(2.0, 1.0)[0]
    aLn = aLT_interpolator(x_a, 1.0)[0]
    kwargs = {
        "Tavol": DEFAULT_MAGS["average_electron_temp"],
        "aLT_core": 2.0,
        "width_axis": x_a,
        "rho": jnp.linspace(0.0, 1.0, 50),
    }
    jax_compatability_test(density_and_temperature_profile_fits.evaluate_profile, fn_kwargs=kwargs)
    kwargs = {
        "Tavol": DEFAULT_MAGS["average_electron_density"],
        "aLT_core": aLn,
        "width_axis": x_a,
        "rho": jnp.linspace(0.0, 1.0, 50),
    }
    jax_compatability_test(density_and_temperature_profile_fits.evaluate_profile, fn_kwargs=kwargs)


def test_operational_limits():
    jax_compatability_test(operational_limits.calc_greenwald_fraction, fn_args=(1.0, 1.0, 1.0, 1.0))
    jax_compatability_test(operational_limits.calc_greenwald_density_limit, fn_args=(1.0, 1.0))
    jax_compatability_test(operational_limits.calc_troyon_limit, fn_args=(1.0, 1.0, 1.0))


def test_reaction_rate_coefficients():
    dummy_ion_temp_profile = jnp.linspace(10.0, 1.0, 50)
    inputs = {"ion_temp_profile": dummy_ion_temp_profile}
    jax_compatability_test(reaction_rate_coefficients.sigmav_DD, fn_kwargs=inputs)
    jax_compatability_test(reaction_rate_coefficients.sigmav_DD_BoschHale, fn_kwargs=inputs)
    jax_compatability_test(reaction_rate_coefficients.sigmav_DHe3, fn_kwargs=inputs)
    jax_compatability_test(reaction_rate_coefficients.sigmav_DT, fn_kwargs=inputs)
    jax_compatability_test(reaction_rate_coefficients.sigmav_DT_BoschHale, fn_kwargs=inputs)
    jax_compatability_test(reaction_rate_coefficients.sigmav_DT_BoschHale, fn_kwargs=inputs)


def test_reaction_energies():
    dummy_ion_temp_profile = jnp.linspace(10.0, 1.0, 50)

    dummy_sigma_v_DT = reaction_rate_coefficients.sigmav_DT(dummy_ion_temp_profile)
    jax_compatability_test(reaction_energies.reaction_energy_DT, fn_args=(dummy_sigma_v_DT, 0.5))

    dummy_sigma_v_DD = reaction_rate_coefficients.sigmav_DD(dummy_ion_temp_profile)
    jax_compatability_test(reaction_energies.reaction_energy_DD, fn_kwargs={"sigmav": dummy_sigma_v_DD})

    dummy_sigma_v_DHe3 = reaction_rate_coefficients.sigmav_DHe3(dummy_ion_temp_profile)
    inputs = {
        "sigmav": dummy_sigma_v_DHe3,
        "heavier_fuel_species_fraction": 0.5,
    }
    jax_compatability_test(reaction_energies.reaction_energy_DHe3, fn_kwargs=inputs)


def test_tau_e_from_Wp():
    # Fake-ish numbers.
    fake_data = {
        "confinement_time_scalar": 1.0,
        "plasma_current": DEFAULT_MAGS["plasma_current"],
        "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
        "average_electron_density": DEFAULT_MAGS["average_electron_density"],
        "major_radius": DEFAULT_MAGS["major_radius"],
        "areal_elongation": 1.8,
        "separatrix_elongation": 1.8,
        "inverse_aspect_ratio": 0.3,
        "fuel_average_mass_number": 1.0,
        "triangularity_psi95": 0.3,
        "separatrix_triangularity": 0.3,
        "plasma_stored_energy": 10,
        "q_star": 3.0,
    }

    read_confinement_scalings()

    for scaling in ConfinementScaling.instances:
        fn = tau_e_from_Wp.get_calc_tau_e_and_P_in_from_scaling(ConfinementScaling.instances[scaling])
        jax_compatability_test(fn, fn_kwargs=fake_data)


def test_density_peaking():
    beta_t = Quantity(0.012, "dimensionless")
    inputs_collisionality = {
        "average_electron_density": DEFAULT_MAGS["average_electron_density"],
        "average_electron_temp": DEFAULT_MAGS["average_electron_temp"],
        "major_radius": DEFAULT_MAGS["major_radius"],
        "z_effective": DEFAULT_MAGS["z_effective"],
    }
    effective_collisionality = density_peaking.calc_effective_collisionality(**inputs_collisionality)
    jax_compatability_test(density_peaking.calc_effective_collisionality, fn_kwargs=inputs_collisionality)
    effective_collisionality_og = formulas_og.plasma_profiles.density_peaking.calc_effective_collisionality(
        average_electron_density=DEFAULT_QUANTITIES["average_electron_density"],
        average_electron_temp=DEFAULT_QUANTITIES["average_electron_temp"],
        major_radius=DEFAULT_QUANTITIES["major_radius"],
        z_effective=DEFAULT_QUANTITIES["z_effective"],
    )
    assert jnp.isclose(effective_collisionality, effective_collisionality_og.magnitude)

    inputs_dens_peaking = {
        "effective_collisionality": effective_collisionality,
        "betaE": beta_t.magnitude,
        "nu_noffset": DEFAULT_MAGS["ion_density_peaking_offset"],
    }
    density_peaking_factor = density_peaking.calc_density_peaking(**inputs_dens_peaking)
    jax_compatability_test(density_peaking.calc_density_peaking, fn_kwargs=inputs_dens_peaking)

    density_peaking_factor_og = formulas_og.plasma_profiles.density_peaking.calc_density_peaking(
        effective_collisionality=effective_collisionality_og,
        beta_toroidal=beta_t,
        nu_noffset=DEFAULT_QUANTITIES["ion_density_peaking_offset"],
    )

    assert jnp.isclose(density_peaking_factor, density_peaking_factor_og.magnitude)


def test_inherent():
    inputs_brems = {
        "electron_density_profile": jnp.linspace(20.0, 1.0, 50),
        "electron_temp_profile": jnp.linspace(20.0, 1.0, 50),
        "z_effective": DEFAULT_MAGS["z_effective"],
        "volume_integrator": _DEFAULT_VOLUME_INTEGRATOR,
    }
    jax_compatability_test(inherent.calc_bremsstrahlung_radiation, fn_kwargs=inputs_brems, static_argnames=["volume_integrator"])
    inputs_sync = {
        "electron_density_profile": inputs_brems["electron_density_profile"],
        "electron_temp_profile": inputs_brems["electron_temp_profile"],
        "major_radius": DEFAULT_MAGS["major_radius"],
        "minor_radius": DEFAULT_MAGS["minor_radius"],
        "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
        "separatrix_elongation": DEFAULT_MAGS["separatrix_elongation"],
        "volume_integrator": _DEFAULT_VOLUME_INTEGRATOR,
    }
    jax_compatability_test(inherent.calc_synchrotron_radiation, fn_kwargs=inputs_sync, static_argnames=["volume_integrator"])


def test_radas():
    class DummyFn(eqx.Module):
        def __init__(self):
            pass

        def __call__(self, x, y):
            return 1.0

    inputs = {
        "electron_temp_profile": jnp.linspace(20.0, 1.0, 50),
        "electron_density_profile": jnp.linspace(20.0, 1.0, 50),
        "impurity_concentration": 1.0,
        "volume_integrator": _DEFAULT_VOLUME_INTEGRATOR,
        "Lz_curve": DummyFn(),
    }
    jax_compatability_test(radas.calc_impurity_radiated_power_radas, fn_kwargs=inputs, static_argnames=["volume_integrator", "Lz_curve"])


def test_current_drive():
    f_shaping_args = {
        "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
        "areal_elongation": DEFAULT_MAGS["areal_elongation"],
        "triangularity_psi95": DEFAULT_MAGS["triangularity_psi95"],
    }

    jax_compatability_test(current_drive.calc_f_shaping, fn_kwargs=f_shaping_args)

    f_shaping = current_drive.calc_f_shaping(**f_shaping_args)
    jax_compatability_test(
        current_drive.calc_plasma_current,
        fn_kwargs={
            "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
            "major_radius": DEFAULT_MAGS["major_radius"],
            "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
            "q_star": 3.0,
            "f_shaping": f_shaping,
        },
    )

    jax_compatability_test(
        current_drive.calc_q_star,
        fn_kwargs={
            "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
            "major_radius": DEFAULT_MAGS["major_radius"],
            "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
            "plasma_current": DEFAULT_MAGS["plasma_current"],
            "f_shaping": f_shaping,
        },
    )

    jax_compatability_test(
        current_drive.calc_ohmic_power,
        fn_kwargs={
            "inductive_plasma_current": DEFAULT_MAGS["plasma_current"],
            "loop_voltage": 1.0,
        },
    )

    jax_compatability_test(current_drive.calc_Spitzer_loop_resistivity, fn_args=(DEFAULT_MAGS["average_electron_temp"],))

    spitzer_loop_resistivity = current_drive.calc_Spitzer_loop_resistivity(DEFAULT_MAGS["average_electron_temp"])

    jax_compatability_test(
        current_drive.calc_resistivity_trapped_enhancement,
        fn_kwargs={"inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"], "definition": 1},
        static_argnames=["definition"],
    )

    neoclassical_loop_resistivity_in = {
        "spitzer_resistivity": spitzer_loop_resistivity,
        "z_effective": DEFAULT_MAGS["z_effective"],
        "trapped_particle_fraction": 0.1,
    }
    jax_compatability_test(current_drive.calc_neoclassical_loop_resistivity, fn_kwargs=neoclassical_loop_resistivity_in)
    neoclassical_loop_resistivity = current_drive.calc_neoclassical_loop_resistivity(**neoclassical_loop_resistivity_in)

    jax_compatability_test(
        current_drive.calc_current_relaxation_time,
        fn_kwargs={
            "major_radius": DEFAULT_MAGS["major_radius"],
            "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
            "areal_elongation": DEFAULT_MAGS["areal_elongation"],
            "average_electron_temp": DEFAULT_MAGS["average_electron_temp"],
            "z_effective": DEFAULT_MAGS["z_effective"],
        },
    )

    jax_compatability_test(
        current_drive.calc_loop_voltage,
        fn_kwargs={
            "major_radius": DEFAULT_MAGS["major_radius"],
            "minor_radius": DEFAULT_MAGS["minor_radius"],
            "inductive_plasma_current": 0.9 * DEFAULT_MAGS["plasma_current"],
            "areal_elongation": DEFAULT_MAGS["areal_elongation"],
            "neoclassical_loop_resistivity": neoclassical_loop_resistivity,
        },
    )

    jax_compatability_test(
        current_drive.calc_bootstrap_fraction,
        fn_kwargs={
            "ion_density_peaking": 40.0,
            "electron_density_peaking": 40.0,
            "temperature_peaking": 40.0,
            "z_effective": DEFAULT_MAGS["z_effective"],
            "q_star": 3.0,
            "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
            "beta_poloidal": 0.012,
        },
    )


def test_calc_fuel_average_mass_number():
    jax_compatability_test(
        average_fuel_ion_mass.calc_fuel_average_mass_number,
        static_argnames=["fusion_reaction"],
        fn_kwargs={
            "fusion_reaction": fusion_rates.ReactionType.DT,
            "heavier_fuel_species_fraction": 0.5,
        },
    )


def test_thermal_calc_gain_factor():
    jax_compatability_test(
        Q_thermal_gain_factor.thermal_calc_gain_factor,
        fn_kwargs={
            "P_fusion": 25,
            "P_launched": 5,
        },
    )


def test_helpers():
    jax_compatability_test(
        helpers.integrate_profile_over_volume_cylindrical,
        fn_kwargs={
            "array_per_m3": jnp.linspace(40, 10, 50),
            "rho": jnp.linspace(0.0, 1.0, 50),
            "plasma_volume": DEFAULT_MAGS["plasma_volume"],
        },
    )

    jax_compatability_test(
        helpers.integrate_profile_over_volume,
        fn_kwargs={
            "array_per_m3": jnp.linspace(40, 10, 50),
            "rho": jnp.linspace(0.0, 1.0, 50),
            "dV_drho": jnp.linspace(0, 10, 50),
        },
    )


def test_figures_of_merit():
    jax_compatability_test(
        figures_of_merit.calc_triple_product,
        fn_kwargs={
            "peak_fuel_ion_density": DEFAULT_MAGS["average_electron_density"],
            "peak_ion_temp": DEFAULT_MAGS["average_ion_temp"],
            "energy_confinement_time": 1,
        },
    )
    jax_compatability_test(
        figures_of_merit.calc_rho_star,
        fn_kwargs={
            "fuel_average_mass_number": DEFAULT_MAGS["fuel_average_mass_number"],
            "average_ion_temp": DEFAULT_MAGS["average_ion_temp"],
            "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
            "minor_radius": DEFAULT_MAGS["minor_radius"],
        },
    )
    jax_compatability_test(
        figures_of_merit.calc_coulomb_logarithm,
        fn_kwargs={
            "ne": DEFAULT_MAGS["average_electron_density"],
            "Te": DEFAULT_MAGS["average_electron_temp"],
        },
    )
    jax_compatability_test(
        figures_of_merit.calc_normalised_collisionality,
        fn_kwargs={
            "average_electron_density": DEFAULT_MAGS["average_electron_density"],
            "average_electron_temp": DEFAULT_MAGS["average_electron_temp"],
            "average_ion_temp": DEFAULT_MAGS["average_ion_temp"],
            "q_star": 3,
            "major_radius": DEFAULT_MAGS["major_radius"],
            "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
            "z_effective": DEFAULT_MAGS["z_effective"],
        },
    )
    jax_compatability_test(
        figures_of_merit.calc_peak_pressure,
        fn_kwargs={
            "peak_electron_temp": DEFAULT_MAGS["average_electron_temp"],
            "peak_ion_temp": DEFAULT_MAGS["average_ion_temp"],
            "peak_electron_density": DEFAULT_MAGS["average_electron_density"],
            "peak_fuel_ion_density": DEFAULT_MAGS["average_electron_density"],
        },
    )


def test_divertor_metrics():
    jax_compatability_test(
        divertor_metrics.calc_B_pol_omp,
        fn_kwargs={
            "plasma_current": DEFAULT_MAGS["plasma_current"],
            "minor_radius": DEFAULT_MAGS["minor_radius"],
        },
    )
    jax_compatability_test(
        divertor_metrics.calc_B_tor_omp,
        fn_kwargs={
            "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
            "minor_radius": DEFAULT_MAGS["minor_radius"],
            "major_radius": DEFAULT_MAGS["major_radius"],
        },
    )


def test_confinement_regime_threshold_powers():
    jax_compatability_test(
        confinement_regime_threshold_powers.calc_LH_transition_threshold_power,
        fn_kwargs={
            "plasma_current": DEFAULT_MAGS["plasma_current"],
            "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
            "minor_radius": DEFAULT_MAGS["minor_radius"],
            "major_radius": DEFAULT_MAGS["major_radius"],
            "surface_area": DEFAULT_MAGS["surface_area"],
            "fuel_average_mass_number": DEFAULT_MAGS["fuel_average_mass_number"],
            "average_electron_density": DEFAULT_MAGS["average_electron_density"],
            "scale": 1.0,
        },
    )
    jax_compatability_test(
        confinement_regime_threshold_powers.calc_LI_transition_threshold_power,
        fn_kwargs={
            "plasma_current": DEFAULT_MAGS["plasma_current"],
            "average_electron_density": DEFAULT_MAGS["average_electron_density"],
            "scale": 1.0,
        },
    )
    for scaling in cfsno.ConfinementPowerScaling:
        if scaling.name not in ["LOC"]:
            jax_compatability_test(
                confinement_regime_threshold_powers.calc_confinement_transition_threshold_power,
                static_argnames=["energy_confinement_scaling"],
                fn_kwargs={
                    "energy_confinement_scaling": scaling,
                    "plasma_current": DEFAULT_MAGS["plasma_current"],
                    "magnetic_field_on_axis": DEFAULT_MAGS["magnetic_field_on_axis"],
                    "minor_radius": DEFAULT_MAGS["minor_radius"],
                    "major_radius": DEFAULT_MAGS["major_radius"],
                    "surface_area": DEFAULT_MAGS["surface_area"],
                    "fuel_average_mass_number": DEFAULT_MAGS["fuel_average_mass_number"],
                    "average_electron_density": DEFAULT_MAGS["average_electron_density"],
                    "confinement_threshold_scalar": 1.0,
                },
            )


DEFAULT_SOL_QUANTITIES = {
    "average_total_pressure": Quantity(732028, "Pa"),
    "power_crossing_separatrix": Quantity(200, "MW"),  # TODO IS THIS REASONABLE?
    "target_electron_temp": Quantity(25.0, "eV"),
    "parallel_heat_flux_density": Quantity(0.02, "GW / m^2"),  # TODO IS THIS REASONABLE?
    "parallel_connection_length": Quantity(30.0, "m"),
    "upstream_electron_density": 0.3 * Quantity(31, "n19"),
    "toroidal_flux_expansion": Quantity(0.6974, "dimensionless"),
    "kappa_e0": Quantity(2600.0, "W / eV^3.5 / m"),
    "sheath_heat_transmission_factor": Quantity(7.5, "dimensionless"),
    "SOL_conduction_fraction": Quantity(1.0, "dimensionless"),
    "target_ratio_of_ion_to_electron_temp": Quantity(1.0, "dimensionless"),
    "target_ratio_of_electron_to_ion_density": Quantity(1.0, "dimensionless"),
    "target_mach_number": Quantity(1.0, "dimensionless"),
    "upstream_ratio_of_ion_to_electron_temp": Quantity(1.0, "dimensionless"),
    "upstream_ratio_of_electron_to_ion_density": Quantity(1.0, "dimensionless"),
    "upstream_mach_number": Quantity(0.0, "dimensionless"),
}
DEFAULT_SOL_MAGS = {k: v.magnitude for k, v in DEFAULT_SOL_QUANTITIES.items()}


def test_scrape_off_layer_model():
    for scaling in cfsno.LambdaQScaling:
        jax_compatability_test(
            lambda_q.calc_lambda_q,
            static_argnames=["lambda_q_scaling"],
            fn_kwargs={
                "lambda_q_scaling": scaling,
                "average_total_pressure": DEFAULT_SOL_MAGS["average_total_pressure"],
                "power_crossing_separatrix": DEFAULT_SOL_MAGS["power_crossing_separatrix"],
                "major_radius": DEFAULT_MAGS["major_radius"],
                "B_pol_omp": 1,
                "inverse_aspect_ratio": DEFAULT_MAGS["inverse_aspect_ratio"],
            },
        )

    jax_compatability_test(
        parallel_heat_flux_density.calc_parallel_heat_flux_density,
        fn_kwargs={
            "power_crossing_separatrix": DEFAULT_SOL_MAGS["power_crossing_separatrix"],
            "fraction_of_P_SOL_to_divertor": 0.6,
            "upstream_major_radius": DEFAULT_MAGS["major_radius"],
            "lambda_q": 0.28,
            "upstream_fieldline_pitch": 2,
        },
    )


def test_two_point_models():
    for momfunc in cfsno.MomentumLossFunction:
        jax_compatability_test(
            solve_target_first_two_point_model,
            static_argnames=["SOL_momentum_loss_function"],
            fn_kwargs={
                "target_electron_temp": DEFAULT_SOL_MAGS["target_electron_temp"],
                "parallel_heat_flux_density": DEFAULT_SOL_MAGS["parallel_heat_flux_density"],
                "parallel_connection_length": DEFAULT_SOL_MAGS["parallel_connection_length"],
                "upstream_electron_density": DEFAULT_SOL_MAGS["upstream_electron_density"],
                "toroidal_flux_expansion": DEFAULT_SOL_MAGS["toroidal_flux_expansion"],
                "fuel_average_mass_number": DEFAULT_MAGS["fuel_average_mass_number"],
                "kappa_e0": DEFAULT_SOL_MAGS["kappa_e0"],
                "SOL_momentum_loss_function": momfunc,
                "sheath_heat_transmission_factor": DEFAULT_SOL_MAGS["sheath_heat_transmission_factor"],
                "SOL_conduction_fraction": DEFAULT_SOL_MAGS["SOL_conduction_fraction"],
                "target_ratio_of_ion_to_electron_temp": DEFAULT_SOL_MAGS["target_ratio_of_ion_to_electron_temp"],
                "target_ratio_of_electron_to_ion_density": DEFAULT_SOL_MAGS["target_ratio_of_electron_to_ion_density"],
                "target_mach_number": DEFAULT_SOL_MAGS["target_mach_number"],
                "upstream_ratio_of_ion_to_electron_temp": DEFAULT_SOL_MAGS["upstream_ratio_of_ion_to_electron_temp"],
                "upstream_ratio_of_electron_to_ion_density": DEFAULT_SOL_MAGS["upstream_ratio_of_electron_to_ion_density"],
                "upstream_mach_number": DEFAULT_SOL_MAGS["upstream_mach_number"],
            },
        )

    for momfunc in cfsno.MomentumLossFunction:
        jax_compatability_test(
            solve_two_point_model,
            static_argnames=["SOL_momentum_loss_function"],
            fn_kwargs={
                "SOL_power_loss_fraction": 0.96,
                "parallel_heat_flux_density": DEFAULT_SOL_MAGS["parallel_heat_flux_density"],
                "parallel_connection_length": DEFAULT_SOL_MAGS["parallel_connection_length"],
                "upstream_electron_density": DEFAULT_SOL_MAGS["upstream_electron_density"],
                "toroidal_flux_expansion": DEFAULT_SOL_MAGS["toroidal_flux_expansion"],
                "fuel_average_mass_number": DEFAULT_MAGS["fuel_average_mass_number"],
                "kappa_e0": DEFAULT_SOL_MAGS["kappa_e0"],
                "SOL_momentum_loss_function": momfunc,
                "sheath_heat_transmission_factor": DEFAULT_SOL_MAGS["sheath_heat_transmission_factor"],
                "SOL_conduction_fraction": DEFAULT_SOL_MAGS["SOL_conduction_fraction"],
                "target_ratio_of_ion_to_electron_temp": DEFAULT_SOL_MAGS["target_ratio_of_ion_to_electron_temp"],
                "target_ratio_of_electron_to_ion_density": DEFAULT_SOL_MAGS["target_ratio_of_electron_to_ion_density"],
                "target_mach_number": DEFAULT_SOL_MAGS["target_mach_number"],
                "upstream_ratio_of_ion_to_electron_temp": DEFAULT_SOL_MAGS["upstream_ratio_of_ion_to_electron_temp"],
                "upstream_ratio_of_electron_to_ion_density": DEFAULT_SOL_MAGS["upstream_ratio_of_electron_to_ion_density"],
                "upstream_mach_number": DEFAULT_SOL_MAGS["upstream_mach_number"],
                "max_iterations": 100,
                "upstream_temp_relaxation": 0.5,
                "target_electron_density_relaxation": 0.5,
                "target_temp_relaxation": 0.5,
                "upstream_temp_max_residual": 1e-2,
                "target_electron_density_max_residual": 1e-2,
                "target_temp_max_residual": 1e-2,
            },
        )
