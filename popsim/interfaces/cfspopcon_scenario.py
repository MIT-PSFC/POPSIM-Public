import os

import cfspopcon
import xarray as xr
from cfspopcon.unit_handling import Quantity

import popsim.cfspopcon_jax as cfsjx
from popsim.enums import FuelSpecies, Impurity, SpeciesContainer


def load_cfspopcon_scenario(case_name: str = "SPARC_PRD"):
    CFSJX_DIR = cfsjx.__file__.split("__init__.py")[0]
    case_path = os.path.join(CFSJX_DIR, f"example_cases/{case_name}")
    input_parameters, algorithm, points = cfspopcon.read_case(case_path)
    return input_parameters, algorithm, points


def load_cfspopcon_scenario_for_comet_mirror(case_name: str = "SPARC_PRD"):
    input_parameters, algorithm, points = load_cfspopcon_scenario(case_name)
    # Strip units away from Pint quantities as we currently don't have Jax<->Pint compatibility.
    for k, v in input_parameters.items():
        if isinstance(v, Quantity):
            input_parameters[k] = v.magnitude
        elif isinstance(v, xr.DataArray):
            # Strip away pint units as they are currently not working with Jax.
            # https://github.com/cfs-energy-internal/POPSIM/issues/3
            input_parameters[k] = v.pint.dequantify()

    algorithm.validate_inputs(input_parameters)

    impurity_types = [Impurity(impurity.value) for impurity in input_parameters["impurities"].dim_species.data]

    species_container = SpeciesContainer(species=[FuelSpecies.Deuterium, FuelSpecies.Tritium, *impurity_types])

    """
    For each species, compute the fraction of its density in terms of the total fuel ion density.
    This is a bit inconsitent with CFSPOPCON: https://github.com/cfs-energy-internal/POPSIM/issues/33
    """
    fuel_concentrations = {
        FuelSpecies.Deuterium: (1.0 - input_parameters["heavier_fuel_species_fraction"]),
        FuelSpecies.Tritium: input_parameters["heavier_fuel_species_fraction"],
    }

    impurity_concentrations = dict(
        zip(
            impurity_types,
            input_parameters["impurities"].values,
        )
    )

    species_concentrations = {**fuel_concentrations, **impurity_concentrations}

    return input_parameters, species_container, species_concentrations
