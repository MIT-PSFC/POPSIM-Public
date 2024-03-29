import chex
import numpy as np

import popsim.algorithms.density as density_model
import popsim.simulators.comet_mirror as cm
from popsim.algorithms.geometry import GeometryCFSPopcon
from popsim.enums import FuelSpecies
from popsim.interfaces.cfspopcon_scenario import load_cfsopcon_scenario


@chex.dataclass
class Sparc2020TestData:
    """Data from Creely et al., 2020. Meant to serve as a reference for testing."""

    R0: float = 1.85  # major radius [m]
    a_minor: float = 0.57  # minor radius [m]
    epsilon: float = 0.31  # inverse aspect ratio [-]
    B0: float = 12.2  # magnetic field [T]
    Ip: float = 8.7  # plasma current [MA]
    kappa_sep: float = 1.97  # elongation at separatrix [-]
    delta_sep: float = 0.54  # triangularity at separatrix [-]
    Q: float = 11.0  # Energy gain factor [-]
    qstar_uckan: float = 3.05  # qstar calculated with the Uckan formula [-]
    rhostar: float = 0.0027  # normalized ion Larmor radius [-]
    nu_eff: float = 0.16  # effective collisionality [-]
    nu_star: float = 0.029  # dimensionless collisionality [-]
    Hfactor: float = 1.0  # H-factor [-]
    tauE: float = 0.77  # energy confinement time [s]
    Paux: float = 11.1  # auxiliary power [MW]
    Pohm: float = 1.7  # ohmic power [MW]
    Zeff: float = 1.5  # effective charge [-]
    dilution: float = 0.85  # dilution [-]
    Te_vol: float = 7.3  # volume-averaged electron temperature [keV]
    Ti_vol: float = 7.3  # volume-averaged ion temperature [keV]
    ne_vol: float = 31.0  # volume-averaged electron density [1e19 m^-3]
    ni_vol: float = 27.0  # volume-averaged ion density [1e19 m^-3]
    nu_Te: float = 2.5  # electron temperature peaking factor [-]
    nu_ni: float = 2.5  # ion density peaking factor [-]
    greenwald_frac: float = 0.37  # Greenwald fraction [-]
    beta: float = 0.012  # plasma beta [-]
    betaN: float = 1.0  # normalized beta [-]
    Pfusion: float = 140  # fusion power [MW]


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
    }

    config = cm.model.Config(
        species=species_container,
        profile_form=input_parameters["profile_form"],
        rho=np.linspace(0, 1, 30),
        energy_confinement_scaling=input_parameters["energy_confinement_scaling"],
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
    )

    state = cm.model.State(
        stored_energy=additional_assumptions["stored_energy"],
        density_state=density_model.State(vol_avg_ion=density_states),
    )

    return model, state, params
