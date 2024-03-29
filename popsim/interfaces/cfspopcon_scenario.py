import os

import cfspopcon
from cfspopcon.unit_handling import Quantity

from popsim import SUBMODULES_DIR
from popsim.enums import FuelSpecies, Impurity, SpeciesContainer


def load_cfsopcon_scenario(case_name: str = "SPARC_PRD"):
    case_path = os.path.join(SUBMODULES_DIR, f"cfspopcon/example_cases/{case_name}")
    input_parameters, algorithm, points = cfspopcon.read_case(case_path)

    # Strip units away from Pint quantities as we currently don't have Jax<->Pint compatibility.
    for k, v in input_parameters.items():
        if isinstance(v, Quantity):
            input_parameters[k] = v.magnitude

    algorithm.validate_inputs(input_parameters)

    impurity_types = [Impurity(impurity.value) for impurity in input_parameters["impurities"].dim_species.data]

    species_container = SpeciesContainer(species=[FuelSpecies.Deuterium, FuelSpecies.Tritium, *impurity_types])

    """Compute the fraction of ions for each species."""
    fuel_concentration = 1.0 - sum(input_parameters["impurities"].values)
    fuel_concentrations = {
        FuelSpecies.Deuterium: (1.0 - input_parameters["heavier_fuel_species_fraction"]) * fuel_concentration,
        FuelSpecies.Tritium: input_parameters["heavier_fuel_species_fraction"] * fuel_concentration,
    }

    impurity_concentrations = dict(
        zip(
            impurity_types,
            input_parameters["impurities"].values,
        )
    )

    species_concentrations = {**fuel_concentrations, **impurity_concentrations}

    return input_parameters, species_container, species_concentrations
