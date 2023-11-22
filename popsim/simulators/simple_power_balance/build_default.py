from cfspopcon.unit_handling import Quantity
from popsim.simulators.simple_power_balance.hello_world import HelloWorldModel, Params, State
from popsim.tests import load_sparc_prd_data_f


def build_default():
    input_parameters, algorithm, points, impurity_types, impurity_concentrations = load_sparc_prd_data_f()
    for k, v in input_parameters.items():
        if isinstance(v, Quantity):
            input_parameters[k] = v.magnitude

    model = HelloWorldModel(
        impurities=impurity_types,
        energy_confinement_scaling=input_parameters["energy_confinement_scaling"],
    )

    params = Params(
        major_radius=input_parameters["major_radius"],
        areal_elongation=input_parameters["areal_elongation"],
        magnetic_field_on_axis=input_parameters["magnetic_field_on_axis"],
        inverse_aspect_ratio=input_parameters["inverse_aspect_ratio"],
        elongation_ratio_sep_to_areal = input_parameters["elongation_ratio_sep_to_areal"],
        triangularity_psi95 = input_parameters["triangularity_psi95"],
        triangularity_ratio_sep_to_psi95 = input_parameters["triangularity_ratio_sep_to_psi95"],
        plasma_current = input_parameters["plasma_current"],
        fraction_of_external_power_coupled = input_parameters["fraction_of_external_power_coupled"],
        heavier_fuel_species_fraction = input_parameters["heavier_fuel_species_fraction"],
        normalized_inverse_temp_scale_length = input_parameters["normalized_inverse_temp_scale_length"],
        electron_density_peaking_offset = input_parameters["electron_density_peaking_offset"],
        ion_density_peaking_offset = input_parameters["ion_density_peaking_offset"],
        temperature_peaking = input_parameters["temperature_peaking"],
        ion_to_electron_temp_ratio = input_parameters["ion_to_electron_temp_ratio"],
        impurity_concentrations=impurity_concentrations,
        confinement_time_scalar=input_parameters["confinement_time_scalar"],
        P_aux_MW=14, # Default value from Creely 2020.
        average_ion_density=27 # From Creely 2020.
    )

    state = State(
        stored_energy=23.0 # Admittedly slightly fudged.
    )

    return model, state, params
