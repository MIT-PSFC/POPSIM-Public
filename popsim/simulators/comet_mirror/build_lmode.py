import numpy as np

import popsim.algorithms.density as density_model
from cfspopcon.unit_handling import Quantity
from popsim.algorithms.geometry import GeometryCFSPopcon
from popsim.enums import FuelSpecies
from popsim.simulators.comet_mirror.model import CometMirror, Config, Params, State
from popsim.tests import load_sparc_q1l_data_f


def build_lmode():
    input_parameters, algorithm, points, impurity_types, impurity_concentrations = load_sparc_q1l_data_f()
    for k, v in input_parameters.items():
        if isinstance(v, Quantity):
            input_parameters[k] = v.magnitude

    config = Config(
        species=[FuelSpecies.Deuterium, FuelSpecies.Tritium, *impurity_types],
        profile_form=input_parameters["profile_form"],
        rho=np.linspace(0, 1, 30),
        energy_confinement_scaling=input_parameters["energy_confinement_scaling"],
    )

    model = CometMirror(config=config)
    geom = GeometryCFSPopcon(
        major_radius=input_parameters["major_radius"],
        inverse_aspect_ratio=input_parameters["inverse_aspect_ratio"],
        areal_elongation=input_parameters["areal_elongation"],
        elongation_ratio_sep_to_areal=input_parameters["elongation_ratio_sep_to_areal"],
        triangularity_psi95=input_parameters["triangularity_psi95"],
        triangularity_ratio_sep_to_psi95=input_parameters["triangularity_ratio_sep_to_psi95"],
    )
    average_ion_density = 13  # From CFSPOPON (could tweak)
    density_states = {
        FuelSpecies.Deuterium: average_ion_density / 2,
        FuelSpecies.Tritium: average_ion_density / 2,
    } | {imp: impurity_concentrations[imp] * average_ion_density for imp in impurity_types}

    particle_confinement_scalars = {k: 3.0 if k in FuelSpecies else 8.0 for k in model.config.species}

    params = Params(
        magnetic_field_on_axis=input_parameters["magnetic_field_on_axis"],
        plasma_current=input_parameters["plasma_current"],
        fraction_of_external_power_coupled=input_parameters["fraction_of_external_power_coupled"],
        normalized_inverse_temp_scale_length=input_parameters["normalized_inverse_temp_scale_length"],
        electron_density_peaking_offset=input_parameters["electron_density_peaking_offset"],
        ion_density_peaking_offset=input_parameters["ion_density_peaking_offset"],
        temperature_peaking=input_parameters["temperature_peaking"],
        ion_to_electron_temp_ratio=input_parameters["ion_to_electron_temp_ratio"],
        confinement_time_scalar=input_parameters["confinement_time_scalar"],
        P_aux_MW=9.4,  # From CFSPOPON (could tweak)
        geometry=geom,
        fueling19={
            FuelSpecies.Deuterium: 150.0,
            FuelSpecies.Tritium: 150.0,
        }
        | {imp: density_states[imp] / particle_confinement_scalars[imp] for imp in impurity_types},
        particle_confinement_scalar=particle_confinement_scalars,
    )
    state = State(
        stored_energy=8,  # From CFSPOPON (could tweak)
        density_state=density_model.State(vol_avg_ion=density_states),
    )
    return model, state, params


if __name__ == "__main__":
    build_lmode()
