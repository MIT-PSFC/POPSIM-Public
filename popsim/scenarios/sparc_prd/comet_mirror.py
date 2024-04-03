import numpy as np

import popsim.physics.density as density_model
import popsim.physics.hmode_dynamics as hmode
import popsim.simulators.comet_mirror as cm
from popsim.enums import FuelSpecies
from popsim.physics.geometry import GeometryCFSPopcon

from .generic import load_cfsopcon_scenario


def build_comet_mirror_config():
    input_parameters, species_container, species_concentrations = load_cfsopcon_scenario("SPARC_PRD")

    # Assumptions that aren't provided by the CFSPOPCON scenario.
    additional_assumptions = {
        "stored_energy": 23.0,  # Admittedly slightly fudged.
        "average_ion_density19": 27,  # From Creely 2020.
        "k_fuel": 3.0,  # Particle confinement scalar for fuel species.
        "k_impurity": 10.0,  # Particle confinement scalar for impurity species.
        "Paux": 11.1,  # MW
        "deuterium_fueling19": 150.0,  # 1e19/s
        "tritium_fueling19": 150.0,  # 1e19/s
        "hmode_transition_characteristic_time": 0.1,  # s
        "hl_threshold_scalar": 0.8,  # Assumed ratio of PHL/PLH
    }

    config = cm.model.Config(
        species=species_container,
        profile_form=input_parameters["profile_form"],
        rho=np.linspace(0, 1, 30),
    )

    model = cm.model.CometMirror(config=config)

    geom = GeometryCFSPopcon(
        major_radius=input_parameters["major_radius"],
        inverse_aspect_ratio=input_parameters["inverse_aspect_ratio"],
        areal_elongation=input_parameters["areal_elongation"],
        elongation_ratio_sep_to_areal=input_parameters["elongation_ratio_sep_to_areal"],
        triangularity_psi95=input_parameters["triangularity_psi95"],
        triangularity_ratio_sep_to_psi95=input_parameters["triangularity_ratio_sep_to_psi95"],
    )

    """
    Use the average ion density + species concentrations to compute the density states.
    """
    density_states = {k: v * additional_assumptions["average_ion_density19"] for k, v in species_concentrations.items()}

    """
    Assumptions for particle confinement scalars.
    """
    particle_confinement_scalars = {
        k: additional_assumptions["k_fuel"] if k in species_container.fuel_species else additional_assumptions["k_impurity"]
        for k in species_container.species
    }

    params = cm.model.Params(
        magnetic_field_on_axis=input_parameters["magnetic_field_on_axis"],
        plasma_current=input_parameters["plasma_current"],
        fraction_of_external_power_coupled=input_parameters["fraction_of_external_power_coupled"],
        normalized_inverse_temp_scale_length=input_parameters["normalized_inverse_temp_scale_length"],
        electron_density_peaking_offset=input_parameters["electron_density_peaking_offset"],
        ion_density_peaking_offset=input_parameters["ion_density_peaking_offset"],
        temperature_peaking=input_parameters["temperature_peaking"],
        ion_to_electron_temp_ratio=input_parameters["ion_to_electron_temp_ratio"],
        confinement_time_scalar=input_parameters["confinement_time_scalar"],
        P_aux_MW=additional_assumptions["Paux"],
        geometry=geom,
        fueling19={
            FuelSpecies.Deuterium: additional_assumptions["deuterium_fueling19"],
            FuelSpecies.Tritium: additional_assumptions["tritium_fueling19"],
        },
        particle_confinement_scalar=particle_confinement_scalars,
        hmode_transition_characteristic_time=additional_assumptions["hmode_transition_characteristic_time"],
        hl_threshold_scalar=additional_assumptions["hl_threshold_scalar"],
    )

    state = cm.model.State(
        stored_energy=additional_assumptions["stored_energy"],
        density_state=density_model.State(vol_avg_ion=density_states),
        hmode_state=hmode.State(hmode=1.0),
    )

    return model, state, params
