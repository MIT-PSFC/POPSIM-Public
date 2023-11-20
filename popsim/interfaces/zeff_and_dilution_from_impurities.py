from collections.abc import Sequence
from typing import Callable

import equinox as eqx
import jax.numpy as jnp

from cfspopcon.formulas import impurity_effects
from popsim import tree_util
from popsim.interfaces.atomic_data import read_atomic_data
from popsim.interfaces.enums import Impurity


class ImpurityCalculator(eqx.Module):
    impurity_type: Impurity
    mean_charge_curve: Callable[[float], float]

    def __init__(self, impurity_type: Impurity, mean_charge_curve: Callable[[float], float]):
        self.impurity_type = impurity_type
        self.mean_charge_curve = mean_charge_curve

    def __call__(
        self,
        average_electron_density: float,
        average_electron_temp: float,
        impurity_concentration: float,
    ) -> dict[str, float]:
        charge_state = impurity_effects.calc_impurity_charge_state_impl(
            average_electron_density, average_electron_temp, self.mean_charge_curve
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
        average_electron_density: float,
        average_electron_temp: float,
        impurity_concentrations: dict[Impurity, float],
        debug: bool = False,
    ):
        impurity_contributions = [
            self.impurity_calculators[impurity](
                average_electron_density,
                average_electron_temp,
                impurity_concentrations[impurity],
            )
            for impurity in impurity_concentrations.keys()
        ]

        tree_of_arrays = tree_util.tree_transpose(impurity_contributions)

        # A bit funky that we add to STARTING_ZEFF but subtract from STARTING_DILUTION,
        # but this is what is implemented in cfspopcon.
        z_effective = self.STARTING_ZEFF +  jnp.sum(tree_of_arrays["change_in_zeff"])
        dilution = self.STARTING_DILUTION - jnp.sum(tree_of_arrays["change_in_dilution"])

        impurity_concentration_values = jnp.array(list(impurity_concentrations.values()))
        summed_impurity_density = jnp.sum(impurity_concentration_values) * average_electron_density
        average_ion_density = dilution * average_electron_density

        outs = {
            "z_effective": z_effective,
            "dilution": dilution,
            "summed_impurity_density": summed_impurity_density,
            "average_ion_density": average_ion_density,
        }

        if debug:
            outs["impurity_contributions"] = impurity_contributions
        return outs
