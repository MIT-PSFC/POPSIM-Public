from cfspopcon.jax_compatible import current_drive


def calc_ohmic_power(
    bootstrap_fraction: float,
    average_electron_temp: float,
    inverse_aspect_ratio: float,
    z_effective: float,
    major_radius: float,
    minor_radius: float,
    areal_elongation: float,
    plasma_current: float,
) -> tuple[float, dict[str, float]]:
    """_summary_

    Args:
        bootstrap_fraction (float): bootstrap fraction [-]
        average_electron_temp (float): average electron temperature [keV]
        inverse_aspect_ratio (float): inverse aspect ratio [-]
        z_effective (float): z effective [-]
        major_radius (float): major radius [m]
        minor_radius (float): minor radius [m]
        areal_elongation (float): area elongation [-]
        plasma_current (float): plasma current [MA]

    Returns:
        Dict[str, float]: _description_
    """
    inductive_plasma_current = plasma_current * (1.0 - bootstrap_fraction)
    spitzer_resistivity = current_drive.calc_Spitzer_loop_resistivity(average_electron_temp)
    trapped_particle_fraction = current_drive.calc_resistivity_trapped_enhancement(inverse_aspect_ratio)
    neoclassical_loop_resistivity = current_drive.calc_neoclassical_loop_resistivity(
        spitzer_resistivity, z_effective, trapped_particle_fraction
    )
    loop_voltage = current_drive.calc_loop_voltage(
        major_radius, minor_radius, inductive_plasma_current, areal_elongation, neoclassical_loop_resistivity
    )
    P_ohmic = current_drive.calc_ohmic_power(inductive_plasma_current, loop_voltage)
    debug = {
        "inductive_plasma_current": inductive_plasma_current,
        "spitzer_resistivity": spitzer_resistivity,
        "trapped_particle_fraction": trapped_particle_fraction,
        "neoclassical_loop_resistivity": neoclassical_loop_resistivity,
        "loop_voltage": loop_voltage,
    }
    return P_ohmic, debug
