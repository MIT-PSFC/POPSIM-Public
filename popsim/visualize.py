import holoviews as hv
import hvplot.xarray  # noqa: F401
import xarray as xr


def visualize_time_series(dataset: xr.Dataset) -> hv.Layout:
    """Visualize the time series of the variables in the dataset.

    Args:
        dataset (xr.Dataset): xarray dataset with dimensions "time" and "simulation".

    Returns:
        hv.Layout: HoloViews layout with the time series of the variables.
    """
    plots = []

    for var in dataset.data_vars:
        # Extract the current variable to plot
        current_var = dataset[var]

        # Creating the line plot for this variable over the time for each simulation
        plot = current_var.hvplot.line(
            x="time",
            by="simulation",
            color="blue",
            legend=False,
            xlabel="Time",
            ylabel=var,
            title=f"{var}",
        )

        # Append the plot to the collection
        plots.append(plot)
    return hv.Layout(plots)
