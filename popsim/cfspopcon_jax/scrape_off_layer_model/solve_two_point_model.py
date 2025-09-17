"""Compute all terms in the two-point-model for a fixed SOL power loss fraction."""

import jax.numpy as np
import xarray as xr
from cfspopcon.named_options import MomentumLossFunction
from cfspopcon.unit_handling import Quantity, Unitfull
from jax import lax

from .momentum_loss_functions import calc_SOL_momentum_loss_function_array, get_SOL_momentum_loss_function
from .target_electron_density import (
    calc_f_other_target_electron_density,
    calc_f_vol_loss_target_electron_density,
    calc_target_electron_density,
    calc_target_electron_density_basic,
)
from .target_electron_flux import (
    calc_f_other_target_electron_flux,
    calc_f_vol_loss_target_electron_flux,
    calc_target_electron_flux,
    calc_target_electron_flux_basic,
)
from .target_electron_temp import (
    calc_f_other_target_electron_temp,
    calc_f_vol_loss_target_electron_temp,
    calc_target_electron_temp,
    calc_target_electron_temp_basic,
)
from .total_pressure import calc_upstream_total_pressure
from .upstream_electron_temp import calc_upstream_electron_temp


def fn_cond(state: dict):
    upstream_electron_temp = state["upstream_electron_temp"]
    target_electron_temp = state["target_electron_temp"]
    target_electron_density = state["target_electron_density"]
    change_in_upstream_electron_temp = state["change_in_upstream_electron_temp"]
    change_in_target_electron_density = state["change_in_target_electron_density"]
    change_in_target_electron_temp = state["change_in_target_electron_temp"]
    upstream_temp_max_residual = state["upstream_temp_max_residual"]
    target_electron_density_max_residual = state["target_electron_density_max_residual"]
    target_temp_max_residual = state["target_temp_max_residual"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    return iteration > max_iterations | np.all(
        np.array(
            [
                np.abs(change_in_upstream_electron_temp / upstream_electron_temp).max() < upstream_temp_max_residual,
                np.abs(change_in_target_electron_density / target_electron_density).max() < target_electron_density_max_residual,
                np.abs(change_in_target_electron_temp / target_electron_temp).max() < target_temp_max_residual,
            ]
        )
    )


def fn_body(state: dict):
    target_electron_temp = state["target_electron_temp"]
    parallel_heat_flux_density = state["parallel_heat_flux_density"]
    parallel_connection_length = state["parallel_connection_length"]
    SOL_conduction_fraction = state["SOL_conduction_fraction"]
    kappa_e0 = state["kappa_e0"]
    upstream_electron_density = state["upstream_electron_density"]
    upstream_ratio_of_ion_to_electron_temp = state["upstream_ratio_of_ion_to_electron_temp"]
    upstream_ratio_of_electron_to_ion_density = state["upstream_ratio_of_electron_to_ion_density"]
    upstream_mach_number = state["upstream_mach_number"]
    SOL_power_loss_fraction = state["SOL_power_loss_fraction"]
    SOL_momentum_array = state["SOL_momentum_array"]
    fuel_average_mass_number = state["fuel_average_mass_number"]
    sheath_heat_transmission_factor = state["sheath_heat_transmission_factor"]
    f_other_target_electron_density = state["f_other_target_electron_density"]
    f_other_target_electron_temp = state["f_other_target_electron_temp"]
    upstream_temp_relaxation = state["upstream_temp_relaxation"]
    target_electron_density_relaxation = state["target_electron_density_relaxation"]
    target_temp_relaxation = state["target_temp_relaxation"]
    upstream_electron_temp = state["upstream_electron_temp"]
    target_electron_temp = state["target_electron_temp"]
    target_electron_density = state["target_electron_density"]
    upstream_temp_max_residual = state["upstream_temp_max_residual"]
    target_electron_density_max_residual = state["target_electron_density_max_residual"]
    target_temp_max_residual = state["target_temp_max_residual"]
    iteration = state["iteration"]
    max_iterations = state["max_iterations"]

    iteration += 1

    new_upstream_electron_temp = calc_upstream_electron_temp(
        target_electron_temp=target_electron_temp,
        parallel_heat_flux_density=parallel_heat_flux_density,
        parallel_connection_length=parallel_connection_length,
        SOL_conduction_fraction=SOL_conduction_fraction,
        kappa_e0=kappa_e0,
    )

    upstream_total_pressure = calc_upstream_total_pressure(
        upstream_electron_density=upstream_electron_density,
        upstream_electron_temp=new_upstream_electron_temp,
        upstream_ratio_of_ion_to_electron_temp=upstream_ratio_of_ion_to_electron_temp,
        upstream_ratio_of_electron_to_ion_density=upstream_ratio_of_electron_to_ion_density,
        upstream_mach_number=upstream_mach_number,
    )

    f_vol_loss_kwargs = dict(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        SOL_momentum_loss_fraction=calc_SOL_momentum_loss_function_array(SOL_momentum_array, target_electron_temp),
    )

    f_basic_kwargs = dict(
        fuel_average_mass_number=fuel_average_mass_number,
        parallel_heat_flux_density=parallel_heat_flux_density,
        upstream_total_pressure=upstream_total_pressure,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )

    target_electron_density_basic = calc_target_electron_density_basic(**f_basic_kwargs)
    target_electron_temp_basic = calc_target_electron_temp_basic(**f_basic_kwargs)

    f_vol_loss_target_electron_density = calc_f_vol_loss_target_electron_density(**f_vol_loss_kwargs)
    f_vol_loss_target_electron_temp = calc_f_vol_loss_target_electron_temp(**f_vol_loss_kwargs)

    new_target_electron_density = calc_target_electron_density(
        target_electron_density_basic=target_electron_density_basic,
        f_vol_loss_target_electron_density=f_vol_loss_target_electron_density,
        f_other_target_electron_density=f_other_target_electron_density,
    )

    new_target_electron_temp = calc_target_electron_temp(
        target_electron_temp_basic=target_electron_temp_basic,
        f_vol_loss_target_electron_temp=f_vol_loss_target_electron_temp,
        f_other_target_electron_temp=f_other_target_electron_temp,
    )

    change_in_upstream_electron_temp = new_upstream_electron_temp - upstream_electron_temp
    change_in_target_electron_density = new_target_electron_density - target_electron_density
    change_in_target_electron_temp = new_target_electron_temp - target_electron_temp

    upstream_electron_temp = upstream_electron_temp + upstream_temp_relaxation * change_in_upstream_electron_temp
    target_electron_density = target_electron_density + target_electron_density_relaxation * change_in_target_electron_density
    target_electron_temp = target_electron_temp + target_temp_relaxation * change_in_target_electron_temp

    state = {
        "target_electron_temp": target_electron_temp,
        "parallel_heat_flux_density": parallel_heat_flux_density,
        "parallel_connection_length": parallel_connection_length,
        "SOL_conduction_fraction": SOL_conduction_fraction,
        "kappa_e0": kappa_e0,
        "upstream_electron_density": upstream_electron_density,
        "upstream_ratio_of_ion_to_electron_temp": upstream_ratio_of_ion_to_electron_temp,
        "upstream_ratio_of_electron_to_ion_density": upstream_ratio_of_electron_to_ion_density,
        "upstream_mach_number": upstream_mach_number,
        "SOL_power_loss_fraction": SOL_power_loss_fraction,
        "SOL_momentum_array": SOL_momentum_array,
        "fuel_average_mass_number": fuel_average_mass_number,
        "sheath_heat_transmission_factor": sheath_heat_transmission_factor,
        "f_other_target_electron_density": f_other_target_electron_density,
        "f_other_target_electron_temp": f_other_target_electron_temp,
        "upstream_temp_relaxation": upstream_temp_relaxation,
        "target_electron_density_relaxation": target_electron_density_relaxation,
        "target_temp_relaxation": target_temp_relaxation,
        "change_in_upstream_electron_temp": change_in_upstream_electron_temp,
        "change_in_target_electron_density": change_in_target_electron_density,
        "change_in_target_electron_temp": change_in_target_electron_temp,
        "upstream_temp_max_residual": upstream_temp_max_residual,
        "target_electron_density_max_residual": target_electron_density_max_residual,
        "target_temp_max_residual": target_temp_max_residual,
        "upstream_electron_temp": upstream_electron_temp,
        "target_electron_density": target_electron_density,
        "iteration": iteration,
        "max_iterations": max_iterations,
    }

    return state


def solve_two_point_model(
    SOL_power_loss_fraction: Unitfull,
    parallel_heat_flux_density: Unitfull,
    parallel_connection_length: Unitfull,
    upstream_electron_density: Unitfull,
    toroidal_flux_expansion: Unitfull,
    fuel_average_mass_number: Unitfull,
    kappa_e0: Unitfull,
    SOL_momentum_loss_function: MomentumLossFunction | xr.DataArray,
    initial_target_electron_temp: float = 10.0,  # eV
    sheath_heat_transmission_factor: float = 7.5,
    SOL_conduction_fraction: float = 1.0,
    target_ratio_of_ion_to_electron_temp: float = 1.0,
    target_ratio_of_electron_to_ion_density: float = 1.0,
    target_mach_number: float = 1.0,
    upstream_ratio_of_ion_to_electron_temp: float = 1.0,
    upstream_ratio_of_electron_to_ion_density: float = 1.0,
    upstream_mach_number: float = 0.0,
    max_iterations: int = 100,
    upstream_temp_relaxation: float = 0.5,
    target_electron_density_relaxation: float = 0.5,
    target_temp_relaxation: float = 0.5,
    upstream_temp_max_residual: float = 1e-2,
    target_electron_density_max_residual: float = 1e-2,
    target_temp_max_residual: float = 1e-2,
) -> tuple[Quantity | xr.DataArray, Quantity | xr.DataArray, Quantity | xr.DataArray, Quantity | xr.DataArray]:
    """Calculate the upstream and target electron temperature and target electron density according to the extended two-point-model.

    Args:
        SOL_power_loss_fraction: [~]
        parallel_heat_flux_density: [GW/m^2]
        parallel_connection_length: [m]
        upstream_electron_density: [10^19 m^-3]
        toroidal_flux_expansion: [~]
        fuel_average_mass_number: [~]
        kappa_e0: electron heat conductivity constant [W / (eV^3.5 * m)]
        SOL_momentum_loss_function: which momentum loss function to use
        initial_target_electron_temp: starting guess for target electron temp [eV]
        sheath_heat_transmission_factor: [~]
        SOL_conduction_fraction: [~]
        target_ratio_of_ion_to_electron_temp: [~]
        target_ratio_of_electron_to_ion_density: [~]
        target_mach_number: [~]
        upstream_ratio_of_ion_to_electron_temp: [~]
        upstream_ratio_of_electron_to_ion_density: [~]
        upstream_mach_number: [~]
        max_iterations: how many iterations to try before returning NaN
        upstream_temp_relaxation: step-size for upstream Te evolution
        target_electron_density_relaxation: step-size for target ne evolution
        target_temp_relaxation: step-size for target Te evolution
        upstream_temp_max_residual: relative rate of change for convergence for upstream Te evolution
        target_electron_density_max_residual: relative rate of change for convergence for target ne evolution
        target_temp_max_residual: relative rate of change for convergence for target Te evolution
    Returns:
        upstream_electron_temp [eV], target_electron_density [m^-3], target_electron_temp [eV], target_electron_flux [m^-2 s^-1]
    """
    f_other_kwargs = dict(
        target_ratio_of_ion_to_electron_temp=target_ratio_of_ion_to_electron_temp,
        target_ratio_of_electron_to_ion_density=target_ratio_of_electron_to_ion_density,
        target_mach_number=target_mach_number,
        toroidal_flux_expansion=toroidal_flux_expansion,
    )
    f_other_target_electron_density = calc_f_other_target_electron_density(**f_other_kwargs)
    f_other_target_electron_temp = calc_f_other_target_electron_temp(**f_other_kwargs)
    f_other_target_electron_flux = calc_f_other_target_electron_flux(**f_other_kwargs)

    iteration = 0
    target_electron_temp = initial_target_electron_temp

    ## DO THE FIRST ITERATION OUT OF THE LOOP

    new_upstream_electron_temp = calc_upstream_electron_temp(
        target_electron_temp=target_electron_temp,
        parallel_heat_flux_density=parallel_heat_flux_density,
        parallel_connection_length=parallel_connection_length,
        SOL_conduction_fraction=SOL_conduction_fraction,
        kappa_e0=kappa_e0,
    )

    upstream_total_pressure = calc_upstream_total_pressure(
        upstream_electron_density=upstream_electron_density,
        upstream_electron_temp=new_upstream_electron_temp,
        upstream_ratio_of_ion_to_electron_temp=upstream_ratio_of_ion_to_electron_temp,
        upstream_ratio_of_electron_to_ion_density=upstream_ratio_of_electron_to_ion_density,
        upstream_mach_number=upstream_mach_number,
    )

    SOL_momentum_array = get_SOL_momentum_loss_function(SOL_momentum_loss_function)

    f_vol_loss_kwargs = dict(
        SOL_power_loss_fraction=SOL_power_loss_fraction,
        SOL_momentum_loss_fraction=calc_SOL_momentum_loss_function_array(SOL_momentum_array, target_electron_temp),
    )

    f_basic_kwargs = dict(
        fuel_average_mass_number=fuel_average_mass_number,
        parallel_heat_flux_density=parallel_heat_flux_density,
        upstream_total_pressure=upstream_total_pressure,
        sheath_heat_transmission_factor=sheath_heat_transmission_factor,
    )

    target_electron_density_basic = calc_target_electron_density_basic(**f_basic_kwargs)
    target_electron_temp_basic = calc_target_electron_temp_basic(**f_basic_kwargs)

    f_vol_loss_target_electron_density = calc_f_vol_loss_target_electron_density(**f_vol_loss_kwargs)
    f_vol_loss_target_electron_temp = calc_f_vol_loss_target_electron_temp(**f_vol_loss_kwargs)

    new_target_electron_density = calc_target_electron_density(
        target_electron_density_basic=target_electron_density_basic,
        f_vol_loss_target_electron_density=f_vol_loss_target_electron_density,
        f_other_target_electron_density=f_other_target_electron_density,
    )

    new_target_electron_temp = calc_target_electron_temp(
        target_electron_temp_basic=target_electron_temp_basic,
        f_vol_loss_target_electron_temp=f_vol_loss_target_electron_temp,
        f_other_target_electron_temp=f_other_target_electron_temp,
    )

    upstream_electron_temp = new_upstream_electron_temp
    target_electron_density = new_target_electron_density
    target_electron_temp = new_target_electron_temp

    change_in_upstream_electron_temp = new_upstream_electron_temp - upstream_electron_temp
    change_in_target_electron_density = new_target_electron_density - target_electron_density
    change_in_target_electron_temp = new_target_electron_temp - target_electron_temp

    ## SETUP THE WHILE LOOP

    state = {
        "target_electron_temp": target_electron_temp,
        "parallel_heat_flux_density": parallel_heat_flux_density,
        "parallel_connection_length": parallel_connection_length,
        "SOL_conduction_fraction": SOL_conduction_fraction,
        "kappa_e0": kappa_e0,
        "upstream_electron_density": upstream_electron_density,
        "upstream_ratio_of_ion_to_electron_temp": upstream_ratio_of_ion_to_electron_temp,
        "upstream_ratio_of_electron_to_ion_density": upstream_ratio_of_electron_to_ion_density,
        "upstream_mach_number": upstream_mach_number,
        "SOL_power_loss_fraction": SOL_power_loss_fraction,
        "SOL_momentum_array": SOL_momentum_array,
        "fuel_average_mass_number": fuel_average_mass_number,
        "sheath_heat_transmission_factor": sheath_heat_transmission_factor,
        "f_other_target_electron_density": f_other_target_electron_density,
        "f_other_target_electron_temp": f_other_target_electron_temp,
        "upstream_temp_relaxation": upstream_temp_relaxation,
        "target_electron_density_relaxation": target_electron_density_relaxation,
        "target_temp_relaxation": target_temp_relaxation,
        "change_in_upstream_electron_temp": change_in_upstream_electron_temp,
        "change_in_target_electron_density": change_in_target_electron_density,
        "change_in_target_electron_temp": change_in_target_electron_temp,
        "upstream_temp_max_residual": upstream_temp_max_residual,
        "target_electron_density_max_residual": target_electron_density_max_residual,
        "target_temp_max_residual": target_temp_max_residual,
        "upstream_electron_temp": upstream_electron_temp,
        "target_electron_density": target_electron_density,
        "iteration": iteration,
        "max_iterations": max_iterations,
    }

    ## NOW START THE WHILE LOOP

    state = lax.while_loop(cond_fun=fn_cond, body_fun=fn_body, init_val=state)

    target_electron_temp = state["target_electron_temp"]
    parallel_heat_flux_density = state["parallel_heat_flux_density"]
    parallel_connection_length = state["parallel_connection_length"]
    SOL_conduction_fraction = state["SOL_conduction_fraction"]
    kappa_e0 = state["kappa_e0"]
    upstream_electron_density = state["upstream_electron_density"]
    upstream_ratio_of_ion_to_electron_temp = state["upstream_ratio_of_ion_to_electron_temp"]
    upstream_ratio_of_electron_to_ion_density = state["upstream_ratio_of_electron_to_ion_density"]
    upstream_mach_number = state["upstream_mach_number"]
    SOL_power_loss_fraction = state["SOL_power_loss_fraction"]
    fuel_average_mass_number = state["fuel_average_mass_number"]
    sheath_heat_transmission_factor = state["sheath_heat_transmission_factor"]
    f_other_target_electron_density = state["f_other_target_electron_density"]
    f_other_target_electron_temp = state["f_other_target_electron_temp"]
    upstream_temp_relaxation = state["upstream_temp_relaxation"]
    target_electron_density_relaxation = state["target_electron_density_relaxation"]
    target_temp_relaxation = state["target_temp_relaxation"]
    upstream_electron_temp = state["upstream_electron_temp"]
    change_in_upstream_electron_temp = state["change_in_upstream_electron_temp"]
    change_in_target_electron_density = state["change_in_target_electron_density"]
    change_in_target_electron_temp = state["change_in_target_electron_temp"]

    target_electron_flux_basic = calc_target_electron_flux_basic(**f_basic_kwargs)
    f_vol_loss_target_electron_flux = calc_f_vol_loss_target_electron_flux(**f_vol_loss_kwargs)

    target_electron_flux = calc_target_electron_flux(
        target_electron_flux_basic=target_electron_flux_basic,
        f_vol_loss_target_electron_flux=f_vol_loss_target_electron_flux,
        f_other_target_electron_flux=f_other_target_electron_flux,
    )

    mask = (
        (np.abs(change_in_upstream_electron_temp / upstream_electron_temp) < upstream_temp_max_residual)
        & (np.abs(change_in_target_electron_density / target_electron_density) < target_electron_density_max_residual)
        & (np.abs(change_in_target_electron_temp / target_electron_temp) < target_temp_max_residual)
    )

    upstream_electron_temp = np.where(mask, upstream_electron_temp, np.nan)  # type:ignore[no-untyped-call]
    target_electron_density = np.where(mask, target_electron_density, np.nan)  # type:ignore[no-untyped-call]
    target_electron_temp = np.where(mask, target_electron_temp, np.nan)  # type:ignore[no-untyped-call]
    target_electron_flux = np.where(mask, target_electron_flux, np.nan)  # type:ignore[no-untyped-call]

    return upstream_electron_temp, target_electron_density, target_electron_temp, target_electron_flux
