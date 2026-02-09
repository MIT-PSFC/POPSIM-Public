import holoviews as hv
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import SymLogNorm


def plot_flux(
    Fx_or_dict: xr.DataArray | dict[str, xr.DataArray], contour_levels: int = 30, normalize: bool = True, title: str = ""
) -> hv.Layout:
    def plot_flux_single(Fx: xr.DataArray, contour_levels: int = 30, normalize: bool = True, title: str = "") -> hv.Layout:
        if normalize:
            Fx = (Fx - Fx.min()) / (Fx.max() - Fx.min())
        heatmap = hv.Image(Fx, ["rx", "zx"]).opts(cmap="viridis", colorbar=True)
        contours = hv.operation.contours(heatmap, levels=contour_levels).opts(color="black")
        out = (heatmap * contours).opts(show_legend=False, aspect="equal", title=title)
        return out

    if isinstance(Fx_or_dict, xr.DataArray):
        return plot_flux_single(Fx_or_dict, contour_levels, normalize, title)
    else:
        plots = {k: plot_flux_single(v, contour_levels, normalize, title=k) for k, v in Fx_or_dict.items()}
        return hv.Layout(list(plots.values())).opts(title=title)


def plot_matrix(da, symlog: bool = False):
    dims = da.dims
    coords_for_dim = {dim: da[dim].values for dim in dims}

    # Determine the number of rows and columns
    num_rows = len(coords_for_dim[dims[0]])
    num_cols = len(coords_for_dim[dims[1]])

    # Scale the figure size based on rows and columns
    scale_factor = 0.4  # Adjust this factor to control the scaling
    fig_width = num_cols * scale_factor
    fig_height = num_rows * scale_factor

    # Create the figure and axis handles
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Plot the data with a diverging colormap centered around zero
    max_mag = max(abs(da.values.min()), abs(da.values.max()))
    if symlog:
        norm = SymLogNorm(linthresh=0.1, linscale=0.5, base=10, vmin=-max_mag, vmax=max_mag)
        cax = ax.imshow(da.values, aspect="auto", cmap="seismic", norm=norm)
    else:
        cax = ax.imshow(da.values, aspect="auto", cmap="seismic", vmin=-max_mag, vmax=max_mag)
    fig.colorbar(cax, ax=ax)  # Add a colorbar to the figure

    # Set the tick positions and labels
    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(coords_for_dim[dims[1]], rotation=60, ha="right")
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(coords_for_dim[dims[0]])

    # Add axis labels
    ax.set_xlabel(dims[1])
    ax.set_ylabel(dims[0])
    ax.grid(True)
    return fig, ax
