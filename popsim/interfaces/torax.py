import multiprocessing as mp
import tracemalloc
from typing import Any, Optional, Union

import equinox as eqx
import jax
import psutil
import xarray as xr
from jaxtyping import Array
from loguru import logger
from torax import output, simulation_app
from torax.config import build_sim
from tqdm import tqdm

import popsim


def monitor_memory():
    """Monitor memory usage."""
    # Get the resident set size (RSS) memory, which does not include swap
    memory = psutil.Process().memory_info().rss / (1024**2)  # Convert to MB
    snapshot = tracemalloc.take_snapshot()
    return memory, snapshot


from popsim.jax_utils import jax_cpu
from popsim.xarray_utils import pytree_to_xarray

"""
Helper functions to run Torax simulations given a Torax config dictionary.
"""


def _run_torax(config: dict[str, Any]) -> xr.Dataset:
    """Given a Torax config dictionary, run a simulation and return the simulation output as an xarray Dataset.

    Args:
        config (dict[str, Any]): A Torax config dictionary. See the Torax documentation for more information.

    Returns:
        xr.Dataset: The simulation output as an xarray Dataset.
    """

    # Convert jax arrays to lists
    config = jax.tree.map(lambda x: x.tolist() if isinstance(x, Array) else x, config)

    sim = build_sim.build_sim_from_config(config)
    geo = sim.geometry_provider(sim.initial_state.t)

    simulation_app.log_to_stdout("Starting simulation.", color=simulation_app.AnsiColors.GREEN)
    sim_outputs = sim.run()
    simulation_app.log_to_stdout("Finished running simulation.", color=simulation_app.AnsiColors.GREEN)
    state_history = output.StateHistory(sim_outputs, sim.source_models)

    dt = state_history.simulation_output_to_xr(geo, sim.file_restart)

    # Convert the DataTree to an xarray Dataset.
    # In the future, as Torax grows in complexity, it will likely make more sense to operate with datatrees.
    datasets = [dt.to_dataset()] + [node.to_dataset() for node in dt.descendants]

    ds = xr.merge(datasets)

    # It's more convienient to work with the normalized rho_cell dimension
    ds = ds.swap_dims({"rho_cell": "rho_cell_norm"})

    # It's more convienient to work with the normalized rho_face dimension
    # Also, since rho_face can be different for each simulation, we'll reset it as a coordinate
    ds = ds.swap_dims({"rho_face": "rho_face_norm"})
    ds = ds.reset_coords("rho_face")

    # Add the config to the dataset
    config_ds = pytree_to_xarray(eqx.filter(config, eqx.is_array_like))

    ds = xr.merge([ds, config_ds])

    return ds


def _safe_run(args):
    """Run a single Torax simulation safely, adding simulation index to results."""
    case, idx = args
    with jax_cpu():  # Assume this context manager forces CPU-only execution
        try:
            result = _run_torax(case)
            return result.assign_coords(simulation=idx).expand_dims("simulation")
        except Exception as e:
            print(f"An error occurred for config index {idx}: {e}")
            return None


def save_result_to_zarr(ds: xr.Dataset, idx: int, output_zarr: str) -> None:
    """Save a single simulation result to the Zarr file."""
    ds = ds.chunk({"simulation": 1})  # Chunk along the simulation dimension
    if idx == 0:  # Initialize the Zarr store on the first result
        ds.to_zarr(output_zarr, mode="w")
        logger.info(f"Initialized Zarr store at {output_zarr} for simulation {idx}")
    else:  # Append subsequent results
        ds.to_zarr(output_zarr, append_dim="simulation")
        logger.info(f"Appended simulation {idx} to Zarr file at {output_zarr}")


def run_torax(
    config: Union[dict[str, Any], list[dict[str, Any]]],
    max_workers: Optional[int] = None,
    output_zarr: Optional[str] = None,
) -> Union[xr.Dataset, None]:
    """
    Given a Torax config dictionary or a list of them, run the simulation(s) and either return the
    output as an xarray Dataset or save results incrementally to the specified Zarr file.

    Args:
        config (list[dict[str, Any]]): A list of Torax config dictionaries. See the Torax documentation for more information.
        max_workers (int, optional): Max number of workers to use for running multiple simulations via multi-processing. Defaults to None, which will use the number of cases or the number of CPUs minus 1, whichever is smaller.
        output_zarr (str, optional): Full path to the Zarr file where simulation results will be stored incrementally. If provided, results are saved to this file, and the function returns None.

    Returns:
        Union[xr.Dataset, None]: The simulation outputs as an xarray Dataset (if no output_zarr is provided),
                                 otherwise None (results are saved to the Zarr file).
    """

    configs = config if isinstance(config, list) else [config]

    if output_zarr and not output_zarr.endswith(".zarr"):
        raise ValueError("The output_zarr argument must specify a path ending with '.zarr'.")

    tracemalloc.start()  # Start memory tracking
    memory_before, _ = monitor_memory()
    logger.info(f"Starting simulation with {len(configs)} configurations.")
    logger.info(f"Initial memory usage: {memory_before:.2f} MB")

    if len(configs) == 1:
        # Single-threaded for one config
        result = _safe_run((configs[0], 0))
        results = [result] if result is not None else None
    else:
        # Multi-threaded for multiple configs
        num_cases = len(configs)
        num_workers = min(num_cases, max(1, mp.cpu_count() - 1))
        num_workers = min(num_workers, max_workers) if max_workers else num_workers

        ctx = mp.get_context("forkserver")
        results = [] if not output_zarr else None
        with ctx.Pool(processes=num_workers) as pool:
            with tqdm(total=num_cases, desc="Processing", unit="task") as pbar:
                for idx, result in enumerate(pool.imap(_safe_run, zip(configs, range(num_cases)))):
                    if result is not None:
                        if output_zarr:
                            save_result_to_zarr(result, idx, output_zarr)
                        else:
                            results.append(result)
                    memory, _ = monitor_memory()
                    logger.info(f"Iteration {idx}: Current memory usage: {memory:.2f} MB")
                    pbar.update(1)

    memory_after, snapshot = monitor_memory()
    tracemalloc.stop()

    logger.info(f"Memory usage after: {memory_after:.2f} MB")
    logger.info("Top memory usage locations:")
    for stat in snapshot.statistics("lineno")[:10]:
        logger.info(stat)

    if results is None:
        return None
    elif len(results) == 0:
        raise ValueError("No valid results were obtained.")
    elif len(results) == 1:
        return results[0]
    else:
        return xr.concat(results, dim="simulation")


def get_sparc_lmode_base_config():
    # Configuration adapted from the Torax iterhybrid_predictor_corrector example.
    # TODO: get a transport expert to improve the configuration.
    CONFIG = {
        "runtime_params": {
            "plasma_composition": {
                # physical inputs
                "main_ion": {"D": 0.5, "T": 0.5},  # (bundled isotope average)
                "Zeff": 1.5,  # needed for qlknn and fusion power
                # effective impurity charge state.
                "Zimp_override": 10,
            },
            "profile_conditions": {
                "Ip_tot": 8.7,  # total plasma current in MA
                # boundary + initial conditions for T and n
                # initial condition ion temperature for r=0 and r=Rmin
                "Ti": {0.0: {0.0: 15.0, 1.0: 0.2}},
                "Ti_bound_right": 0.2,  # Rodriguez-Fernandez 2024. [keV]
                # initial condition electron temperature for r=0 and r=Rmin
                "Te": {0.0: {0.0: 15.0, 1.0: 0.2}},
                "Te_bound_right": 0.2,  # Rodriguez-Fernandez 2024. [keV]
                "ne_bound_right": 2.0,  # Dummy boundary condition. [1e20]
                "ne_is_fGW": False,
                "ne": {0: {0.0: 5.0, 1.0: 2.0}},  # Initial electron density profile
                "set_pedestal": True,
            },
            "numerics": {
                # simulation control
                "t_final": 5,  # length of simulation time in seconds
                "ion_heat_eq": True,
                "el_heat_eq": True,
                "current_eq": True,
                "dens_eq": True,
                "maxdt": 0.02,
                # multiplier in front of the base timestep dt=dx^2/(2*chi). Can
                # likely be increased further beyond this default.
                "dtmult": 50,
                "dt_reduction_factor": 3,
            },
        },
        "pedestal": {
            "Tiped": 1.0,  # ion pedestal top temperature in keV for Ti and Te
            "Teped": 1.0,  # electron pedestal top temperature in keV for Ti and Te
            "neped": 2.0,  # pedestal top electron density in units of nref
            "rho_norm_ped_top": 0.95,
        },
        "geometry": {
            "geometry_type": "circular",
            "elongation_LCFS": 1.97,  # elongation
            "Rmaj": 1.85,  # major radius (R) in meters
            "Rmin": 0.57,  # minor radius (a) in meters
            "B0": 12.2,  # Toroidal magnetic field on axis [T]
        },
        "sources": {
            # Current sources (for psi equation)
            "j_bootstrap": {
                # Multiplication factor for bootstrap current.
                "bootstrap_mult": 1.0,
            },
            # Electron density sources/sink (for the ne equation).
            "generic_particle_source": {
                # total particle source
                "S_tot": 3.0e21,
                # exponential decay length in rho.
                "puff_decay_length": 0.05,
            },
            # Ion and electron heat sources (for the temp-ion and temp-el eqs).
            "generic_ion_el_heat_source": {
                "rsource": 0.25,
                # Gaussian width in normalized radial coordinate r
                "w": 0.1,
                # total heating (including accounting for radiation) r
                "Ptot": 11.0e6,
                # electron heating fraction r
                "el_heat_fraction": 0.22,  # external power electron heating fraction (Rodriguez-Fernandez, 2020)
            },
            "fusion_heat_source": {},
            "ohmic_heat_source": {},
            "qei_source": {
                # multiplier for ion-electron heat exchange term for sensitivity
                "Qei_mult": 1.0,
            },
        },
        "transport": {
            "transport_model": "qlknn",
            # set inner core transport coefficients (ad-hoc MHD/EM transport)
            "apply_inner_patch": True,  # Disable inner patch transport
            "De_inner": 0.1,
            "Ve_inner": 0.0,
            "chii_inner": 0.25,
            "chie_inner": 0.25,
            "rho_inner": 0.2,  # radius below which patch transport is applied
            # set outer core transport coefficients (L-mode near edge region)
            "apply_outer_patch": True,
            "De_outer": 0.1,
            "Ve_outer": 0.0,
            "chii_outer": 2.0,
            "chie_outer": 2.0,
            "rho_outer": 0.9,  # radius above which patch transport is applied
            # allowed chi and diffusivity bounds
            "chimin": 0.05,  # minimum chi
            "chimax": 100,  # maximum chi (can be helpful for stability)
            "Demin": 0.05,  # minimum electron diffusivity
            "qlknn_inputs": {
                "DVeff": True,
                "include_ITG": True,  # to toggle ITG modes on or off
                "include_TEM": True,  # to toggle TEM modes on or off
                "include_ETG": True,  # to toggle ETG modes on or off
                # ensure that smag - alpha > -0.2 always, to compensate for no slab
                # modes
                "avoid_big_negative_s": True,
                # minimum |R/Lne| below which effective V is used instead of
                # effective D
                "An_min": 0.05,
                "ITG_flux_ratio_correction": 1,
                "model_path": popsim.TORAX_QLKNN_MODEL_PATH,
            },
        },
        "stepper": {
            "stepper_type": "linear",
            "predictor_corrector": True,
            "corrector_steps": 1,
            "use_pereverzev": True,
        },
        "time_step_calculator": {
            "calculator_type": "fixed",
        },
    }
    return CONFIG
