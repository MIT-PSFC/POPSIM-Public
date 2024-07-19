import typing
from collections.abc import Sequence
from typing import Optional, Union

import holoviews as hv
import hvplot.xarray  # noqa: F401
import numpy as np
import panel as pn
import xarray as xr
from jaxtyping import PyTree

import popsim.xarray_utils as pxr
from popsim import param_utils


def visualize_time_series(
    dataset: xr.Dataset,
    plot_spec: Optional[list[Union[str, Sequence[str]]]] = None,
    hlines: Optional[dict[str, float]] = None,
    max_cols: int = 3,
    fontsize: int = 10,
) -> pn.panel:
    """Visualize the time series of the variables in the dataset.

    Args:
        dataset (xr.Dataset): xarray dataset with dimensions "time" and "simulation".
        plot_spec (Optional[list[Union[str, Sequence[str]]]]): List of variables or groups of variables to plot.
        hlines (Optional[dict[str, float]]): Dictionary of horizontal lines to add to plots.
        max_cols (int): Maximum number of columns in the layout.
        fontsize (int): Font size for plot labels and titles.

    Returns:
        pn.panel: Panel containing the time series plots.
    """
    if hlines is None:
        hlines = {}

    plots = []
    variables_to_plot = plot_spec if plot_spec is not None else list(dataset.data_vars)

    for i, var in enumerate(variables_to_plot):
        plot = create_plot(dataset, var, i == 0, fontsize)

        # Add horizontal line if specified
        if any(v in hlines for v in (var if isinstance(var, Sequence) else [var])):
            plot = add_hlines(plot, var, hlines, dataset)

        plots.append(plot)

    layout = hv.Layout(plots).cols(max_cols)
    return pn.panel(layout, sizing_mode="stretch_width")


def create_plot(dataset: xr.Dataset, var: Union[str, Sequence[str]], show_legend: bool, fontsize: int) -> hv.Element:
    """Create a plot for single variable or multiple variables."""
    if isinstance(var, str):
        variables = [var]
        title = var
    else:
        variables = var
        title = ", ".join(variables)

    plots = []
    for v in variables:
        current_var = dataset[v]
        if current_var.dtype == bool:
            current_var = current_var.astype(int)

        plot = current_var.hvplot.line(
            x="time",
            cmap="viridis",
            legend=show_legend,
            xlabel="Time",
            by="simulation" if "simulation" in dataset.dims else None,
        )
        plots.append(plot)

    if len(plots) == 1:
        return plots[0].opts(title=title, ylabel="", fontsize=fontsize)
    else:
        return hv.Overlay(plots).opts(title=title, ylabel="", fontsize=fontsize)


def add_hlines(plot: hv.Element, var: Union[str, Sequence[str]], hlines: dict[str, float], dataset: xr.Dataset) -> hv.Overlay:
    """Add horizontal lines to the plot."""
    variables = [var] if isinstance(var, str) else var
    hline_plots = []
    for v in variables:
        if v in hlines:
            hline = hv.Curve([(dataset.time.min(), hlines[v]), (dataset.time.max(), hlines[v])]).opts(color="red")
            hline_plots.append(hline)
    if hline_plots:
        return hv.Overlay([plot, *hline_plots]).opts(shared_axes=False)
    return plot


def visualize_params(params: typing.Union[PyTree, typing.Sequence[PyTree]], time_base: np.ndarray, interp_type: str = "linear") -> pn.panel:
    """Visualize the params of the simulation. This builds it into vectorized form and converts it to a xr.Dataset for visualization.

    Args:
        params (typing.Union[PyTree, typing.Sequence[PyTree]]): params tree.
        time_base (np.ndarray): Time base for the simulation.
        interp_type (str): Interpolation type.

    Returns:
        pn.panel: panel showing the params as a time trace.
    """
    params_vec, multi_sim = param_utils.build_vectorized_params(params, time_base, interp_type)
    dataset = pxr.time_and_pytree_to_xarray(time_base, params_vec, multi_simulation=multi_sim)
    return visualize_time_series(dataset)
