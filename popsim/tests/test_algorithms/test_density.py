from popsim.enums import FuelSpecies, Impurity


def test_multi_species_density_model():
    fuel_species = [FuelSpecies.Deuterium, FuelSpecies.Tritium]
    impurity_species = [Impurity.Helium, Impurity.Tungsten]
