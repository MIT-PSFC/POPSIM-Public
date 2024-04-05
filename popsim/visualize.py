from typing import Optional

import holoviews as hv
import hvplot.xarray  # noqa: F401
import panel as pn
import xarray as xr

import popsim.config as pconfig
import popsim.xarray_utils as pxr


def visualize_time_series(
    dataset: xr.Dataset, hlines: Optional[dict[str, float]] = None, max_cols: int = 3, fontsize: int = 10
) -> hv.Layout:
    """Visualize the time series of the variables in the dataset.

    Args:
        dataset (xr.Dataset): xarray dataset with dimensions "time" and "simulation".

    Returns:
        hv.Layout: HoloViews layout with the time series of the variables.
    """
    if hlines is None:
        hlines = {}
    plots = []

    for i, var in enumerate(dataset.data_vars):
        # Extract the current variable to plot
        current_var = dataset[var]

        if current_var.dtype == bool:
            # Cast bools to ints for visualization.
            current_var = current_var.astype(int)

        # Creating the line plot for this variable over the time for each simulation
        has_legend = i == 0  # Only show one legend.
        plot = current_var.hvplot.line(
            x="time",
            cmap="viridis",
            legend=has_legend,
            xlabel="Time",
            by="simulation" if "simulation" in dataset.dims else None,  # Separate by simulation if available.
            title=f"{var}",
        ).opts(ylabel="", fontsize=fontsize)

        if var in hlines:
            hline = hv.Curve([(dataset.time.min(), hlines[var]), (dataset.time.max(), hlines[var])]).opts(color="red")
            # For some reason, the overlay causes all plots to be linked unless we set shared_axes=False.
            plot = hv.Overlay([plot, hline]).opts(shared_axes=False)

        plots.append(plot)

    layout = hv.Layout(plots).cols(max_cols)
    return pn.panel(layout, sizing_mode="stretch_width")


def visualize_config(config, time_base, interp_type: str = "linear"):
    """Visualize the configuration of the simulation.

    Args:
        config (dict): Configuration dictionary.
        time_base (np.ndarray): Time base for the simulation.
        interp_type (str): Interpolation type.

    Returns:
        pn.pane.Markdown: Markdown pane with the configuration information.
    """
    config_vec, multi_sim = pconfig.build_vectorized_configs(config, time_base, interp_type)
    dataset = pxr.time_and_pytree_to_xarray(time_base, config_vec, multi_simulation=multi_sim)
    return visualize_time_series(dataset)
