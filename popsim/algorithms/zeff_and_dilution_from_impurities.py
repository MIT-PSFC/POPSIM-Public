from collections.abc import Sequence
from typing import Callable, Dict

import equinox as eqx
import optimistix as optx
import jax.numpy as jnp

from cfspopcon.jax_compatible import impurity_effects
from popsim import tree_util
from popsim.enums import Impurity
from popsim.interfaces.atomic_data import read_atomic_data


class ImpurityCalculator(eqx.Module):
    impurity_type: Impurity
    mean_charge_curve: Callable[[float], float]

    def __init__(self, impurity_type: Impurity, mean_charge_curve: Callable[[float], float]):
        self.impurity_type = impurity_type
        self.mean_charge_curve = mean_charge_curve

    def __call__(
        self,
        average_electron_density_19: float,
        average_electron_temp_keV: float,
        impurity_concentration: float,
    ) -> dict[str, float]:
        charge_state = impurity_effects.calc_impurity_charge_state_impl(
            1e19 * average_electron_density_19, 1000 * average_electron_temp_keV, self.mean_charge_curve
        )
        change_in_zeff = impurity_effects.calc_change_in_zeff(charge_state, impurity_concentration)
        change_in_dilution = impurity_effects.calc_change_in_dilution(charge_state, impurity_concentration)
        out = {
            "charge_state": charge_state,
            "change_in_zeff": change_in_zeff,
            "change_in_dilution": change_in_dilution,
        }
        return out


class CalcZeffAndDilutionFromImpurities(eqx.Module):
    impurity_calculators: dict[Impurity, ImpurityCalculator]
    STARTING_ZEFF = 1.0
    STARTING_DILUTION = 1.0

    def __init__(self, impurities: Sequence[Impurity]):
        atomic_data = read_atomic_data()

        self.impurity_calculators = {
            impurity: ImpurityCalculator(
                impurity,
                atomic_data[impurity].coronal_mean_Z_interpolator,
            )
            for impurity in impurities
        }

    def __call__(
        self,
        average_electron_density_19: float,
        average_electron_temp_keV: float,
        impurity_concentrations: dict[Impurity, float],
        debug: bool = False,
    ):
        impurity_contributions = [
            self.impurity_calculators[impurity](
                average_electron_density_19,
                average_electron_temp_keV,
                impurity_concentrations[impurity],
            )
            for impurity in impurity_concentrations.keys()
        ]

        tree_of_arrays = tree_util.tree_transpose(impurity_contributions)

        # A bit funky that we add to STARTING_ZEFF but subtract from STARTING_DILUTION,
        # but this is what is implemented in cfspopcon.
        z_effective = self.STARTING_ZEFF + jnp.sum(tree_of_arrays["change_in_zeff"])
        dilution = self.STARTING_DILUTION - jnp.sum(tree_of_arrays["change_in_dilution"])

        impurity_concentration_values = jnp.array(list(impurity_concentrations.values()))
        summed_impurity_density = jnp.sum(impurity_concentration_values) * average_electron_density_19
        average_ion_density = dilution * average_electron_density_19

        outs = {
            "z_effective": z_effective,
            "dilution": dilution,
            "summed_impurity_density": summed_impurity_density,
            "average_ion_density": average_ion_density,
        }

        if debug:
            outs["impurity_contributions"] = impurity_contributions
        return outs


class CalcTempDensBreakdown(eqx.Module):
    zeff_and_dilution_calc: CalcZeffAndDilutionFromImpurities
    solver: optx.AbstractFixedPointSolver

    def __init__(
        self,
        zeff_and_dilution_calc: CalcZeffAndDilutionFromImpurities,
        solver: optx.AbstractFixedPointSolver = optx.BFGS(rtol=1e-4, atol=1e-4),
    ):
        self.zeff_and_dilution_calc = zeff_and_dilution_calc
        self.solver = solver

    def __call__(
        self,
        average_pressure: float,
        average_ion_density: float,
        ion_to_electron_temp_ratio: float,
        impurity_concentrations: dict[Impurity, float],
    ) -> Dict[str, float]:
        """Solve for the breakdown of electron + ion densities and temperatures.
        This module makes two major assumptions in its calculations:
            1) Dilution is insensitive to 10s of percent change in average electron density + pressure.
            2) <p> = <n_e> <T_e> + <n_i> ion_to_electron_temp_ratio * <T_e>

        Args:
            average_pressure (float): volume-average pressure [keV * 1e19 m^-3]
            average_ion_density (float): volume-average ion density [1e19 m^-3]
            ion_to_electron_temp_ratio (float): ratio of ion to electron temperature [-]
            impurity_concentrations (dict[Impurity, float]):

        Returns:
            _type_: _description_
        """

        # ASSUMPTION: Dilution is relatively insensitive to electron density and pressure.
        # Spot check suggests this is true.
        """
        ASSUMPTION 1: Dilution is insensitive to 10s of percent change in average electron density + pressure.
        Spot check suggests this is true. Thus, lets use the ion density to guess the electron density for
        the purposes of calculating the dilution.
        """
        average_electron_density_guess = average_ion_density
        average_electron_temp_guess = (0.5 * average_pressure) / average_electron_density_guess
        impurity_outs = self.zeff_and_dilution_calc(
            average_electron_density_19=average_electron_density_guess,
            average_electron_temp_keV=average_electron_temp_guess,
            impurity_concentrations=impurity_concentrations,
        )

        dilution = impurity_outs["dilution"]
        average_electron_density = average_ion_density / dilution

        """
        ASSUMPTION 2:
            <p> = <n_e> <T_e> + <n_i> ion_to_electron_temp_ratio * <T_e>
        """
        average_electron_temp = average_pressure / (average_electron_density + average_ion_density * ion_to_electron_temp_ratio)

        average_ion_temp = ion_to_electron_temp_ratio * average_electron_temp

        """
        Now that we have proper electron temps + densities, we can recalculate the impurity contributions.
        """
        impurity_outs = self.zeff_and_dilution_calc(
            average_electron_density_19=average_electron_density,
            average_electron_temp_keV=average_electron_temp,
            impurity_concentrations=impurity_concentrations,
        )

        outs = {
            "average_electron_density": average_electron_density,
            "average_electron_temp": average_electron_temp,
            "average_ion_density": average_ion_density,
            "average_ion_temp": average_ion_temp,
            "z_effective": impurity_outs["z_effective"],
            "dilution": impurity_outs["dilution"],
            "summed_impurity_density": impurity_outs["summed_impurity_density"],
        }
        return outs
