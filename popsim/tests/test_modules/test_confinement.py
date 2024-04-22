from popsim.modules.confinement import Confinement
from popsim.physics.geometry import GeometryCFSPopcon
import jax.numpy as jnp
import jax
import diffrax
from popsim.simulators.scenario_data.sparc_prd import load_cfspopcon_scenario_for_comet_mirror
from popsim.interp import resolve_paths
from popsim.xarray_utils import solution_to_xarray

def test_dynamics():
    input_parameters, species_container, species_concentrations = load_cfspopcon_scenario_for_comet_mirror("SPARC_PRD")
    # Assumptions that aren't provided by the CFSPOPCON scenario.
    
    additional_assumptions = {
        "stored_energy": 23.0,  # [MJ]
        "average_ion_density19": 27,  # From Creely 2020.
        "k_fuel": 3.0,  # Particle confinement scalar for fuel species.
        "k_impurity": 10.0,  # Particle confinement scalar for impurity species.
        "Paux": 11.1,  # MW
        "deuterium_fueling19": 150.0,  # 1e19/s
        "tritium_fueling19": 150.0,  # 1e19/s
        "hmode_transition_characteristic_time": 0.1,  # s
        "hl_threshold_scalar": 0.8,  # Assumed ratio of PHL/PLH
    }
    confinement_module = Confinement(config=Confinement.Config())
    state = Confinement.State(hmode_state=confinement_module.hmode_dynamics.State(hmode=0.0))


    #
    # Four phases:
    #   1) Conducted power well above LH threshold.
    #   2) Conducted power above HL threshold but below LH threshold.
    #   3) Conducted power well below HL threshold.
    #   4) Conducted power above LH threshold, but input power below LH threshold.
    #
    times = jnp.array([0.0, 0.5, 0.6, 1.0, 1.01, 1.11, 1.3])
    conducted_powers = jnp.array([30.0, 30.0, 27.0, 27.0, 20.0, 20.0, 30.0])
    conducted_powers_traj = diffrax.LinearInterpolation(ts=times, ys=conducted_powers)
    input_powers = jnp.array([30.0, 30.0, 30.0, 30.0, 30.0, 0.0, 0.0])
    input_powers_traj = diffrax.LinearInterpolation(ts=times, ys=input_powers)
    
    geom = GeometryCFSPopcon(
        major_radius=input_parameters["major_radius"],
        inverse_aspect_ratio=input_parameters["inverse_aspect_ratio"],
        areal_elongation=input_parameters["areal_elongation"],
        elongation_ratio_sep_to_areal=input_parameters["elongation_ratio_sep_to_areal"],
        triangularity_psi95=input_parameters["triangularity_psi95"],
        triangularity_ratio_sep_to_psi95=input_parameters["triangularity_ratio_sep_to_psi95"],
    )

    particle_confinement_scalars = {
        k: additional_assumptions["k_fuel"] if k in species_container.fuel_species else additional_assumptions["k_impurity"]
        for k in species_container.species
    }
    
    params = Confinement.Params(
        magnetic_field_on_axis=input_parameters["magnetic_field_on_axis"],  # [T]
        plasma_current=input_parameters["plasma_current"],  # [A]
        stored_energy=additional_assumptions['stored_energy'], # [MJ]
        average_electron_density_19=30, # [1e19/m^3]
        confinement_time_scalar=input_parameters["confinement_time_scalar"],  # [-]
        geometry=geom,
        particle_confinement_scalar=particle_confinement_scalars,  # [-]
        hmode_transition_characteristic_time=additional_assumptions["hmode_transition_characteristic_time"],  # [s]
        hl_threshold_scalar=additional_assumptions["hl_threshold_scalar"],  # Assume the h->l transition is some fraction of the l->h transition [-]
        heavier_fuel_species_fraction=0.5,  # [-]
        fuel_average_mass_number=2.5, #[amu]
        q_star=4,  # [-]
        transition_characteristic_time=0.1,  # amount of time for a L->H or H->L transition to occur [s].
        P_tau_MW = conducted_powers_traj,  # Power conducted to the scrape-off layer [MW]
        P_input_MW = input_powers_traj,  # Power input to the plasma [MW]
)



    def fun(t, y, args, return_aux: bool = False):
        params_t = resolve_paths(args, t)
        state_dot, out = confinement_module(y, params_t)
        if return_aux:
            return out
        else:
            return state_dot

    # Function to save auxiliary information.
    def saveat_fn(t, y, args):
        out = {"state": y, "aux": fun(t, y, args, return_aux=True)}
        return out

    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(fun),
        solver=diffrax.Tsit5(),
        t0=times[0],
        t1=times[-1],
        dt0=jnp.min(jnp.diff(times)),
        y0=state,
        args=params,
        saveat=diffrax.SaveAt(ts=times, fn=saveat_fn),
    )

    dataset = solution_to_xarray(sol, multi_simulation=False)

    assert jnp.isclose(float(dataset['aux.tau_E'][0]),0.1834906)
    assert jnp.isclose(float(dataset['aux.tau_E'][1]),0.67667817)

    assert jnp.isclose(float(dataset['aux.species_confinement_time.Impurity.Tungsten'][0]),1.834906)
    assert jnp.isclose(float(dataset['aux.species_confinement_time.Impurity.Tungsten'][1]),6.7667817)

    in_hmodes = dataset['aux.in_hmode']
    assert in_hmodes[0] == False
    assert in_hmodes[1] == True
    assert in_hmodes[2] == True
    assert in_hmodes[3] == True
    assert in_hmodes[4] == True
    assert in_hmodes[5] == False
    assert in_hmodes[6] == False

