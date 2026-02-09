import jax
import jax.numpy as jnp

from popsim.math_utils import padded_relative_error


def vec_compute_padded_relative_errors(p, t):
    fn = jax.vmap(padded_relative_error, in_axes=(0, 0))
    return fn(p, t)


def eval_padded_relative_errors(eval_data):
    input_ds, output_ds = eval_data.input_ds.transpose("sample", ...), eval_data.output_ds.transpose("sample", ...)

    errors = {
        "Fx": vec_compute_padded_relative_errors(output_ds.Fx.data, input_ds["LY.Fx"].data),
        "Ip": vec_compute_padded_relative_errors(output_ds.Ip.data, input_ds["LY.Ip"].data),
        "bp": vec_compute_padded_relative_errors(output_ds.betap.data, input_ds["LY.bp"].data),
    }
    return errors


def eval_padded_relative_error_medians(eval_data):
    errors = eval_padded_relative_errors(eval_data)
    median_errors = {k: jnp.median(v) for k, v in errors.items()}
    return median_errors


def eval_padded_relative_errors_hist(eval_data):
    import matplotlib.pyplot as plt

    errors = eval_padded_relative_errors(eval_data)
    # Three histograms.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, key in enumerate(errors.keys()):
        ax = axes[i]
        ax.hist(errors[key], bins=200, range=(-0.5, 0.5))
        ax.set_title(f"Relative Error {key}")
        ax.set_xlabel("Relative error")
        ax.set_ylabel("Counts")
        ax.set_xlim(0.0, 0.2)
    return fig


def flux_compare(eval_data):
    import holoviews as hv
    import numpy as np

    from popsim.modules.meqml.visualize import plot_flux

    hv.extension("bokeh")
    times = eval_data.input_ds.time_idx.values
    shots = eval_data.input_ds.shot.values

    def plot_selection(time_idx, shot):
        inp, pred = (
            eval_data.input_ds.sel(time_idx=time_idx, shot=shot).squeeze().drop_vars("sample"),
            eval_data.output_ds.sel(time_idx=time_idx, shot=shot).squeeze().drop_vars("sample"),
        )
        relative_error = np.linalg.norm(pred["Fx"] - inp["LY.Fx"]) / np.linalg.norm(inp["LY.Fx"])
        flux_plot = plot_flux({"Target": inp["LY.Fx"], f"Predicted, Rel Error {relative_error:.5f}": pred["Fx"]})

        diags = (
            inp["LX.Bm"].hvplot.scatter(x="mag_probe", title="Magnetic Probes").opts(width=600)
            + inp["LX.Ff"].hvplot.scatter(x="flux_loop", title="Flux Loops (Ff)").opts(width=600)
            + inp["LX.Uf"].hvplot.scatter(x="flux_loop", title="Flux Loops (Uf)").opts(width=600)
            + inp["LX.Ia"].hvplot.scatter(x="active_coils", title="Active Coils").opts(width=600)
        ).cols(1)
        # Place flux_plot and diags side by side
        return (flux_plot + diags).cols(2)

    dmap_heatmap_contour = hv.DynamicMap(plot_selection, kdims=["time_idx", "shot"]).redim.values(time_idx=times, shot=shots)
    return dmap_heatmap_contour
