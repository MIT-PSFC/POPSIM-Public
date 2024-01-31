from popsim.algorithms.density import (GenericDensityModel,
                                       MultiSpeciesDensityModel)
from popsim.enums import FuelSpecies, Impurity, Species


def test_multi_species_density_model():
    fuel_species = [FuelSpecies.Deuterium, FuelSpecies.Tritium]
    impurity_species = [Impurity.Helium, Impurity.Tungsten]
    model = MultiSpeciesDensityModel(fuel_species + impurity_species)