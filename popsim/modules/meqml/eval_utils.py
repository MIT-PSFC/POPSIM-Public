import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np

from popsim.ml.eval import EvalData
from popsim.modules.meqml.visualize import plot_flux


def wrap_plot_fn_shot(input_ds, output_ds, plot_fn):
    shots = input_ds.shot.values

    def plot_selection(shot):
        inp, pred = (
            input_ds.sel(shot=shot).squeeze(),
            output_ds.sel(shot=shot).squeeze(),
        )
        return plot_fn(pred, inp)

    dmap = hv.DynamicMap(plot_selection, kdims=["shot"]).redim.values(shot=shots)
    return dmap


def wrap_plot_fn(input_ds, output_ds, plot_fn):
    times = input_ds.time_idx.values
    shots = input_ds.shot.values

    def plot_selection(time_idx, shot):
        inp, pred = (
            input_ds.sel(time_idx=time_idx, shot=shot).squeeze().drop_vars("sample"),
            output_ds.sel(time_idx=time_idx, shot=shot).squeeze().drop_vars("sample"),
        )
        return plot_fn(pred, inp)

    dmap = hv.DynamicMap(plot_selection, kdims=["time_idx", "shot"]).redim.values(time_idx=times, shot=shots)
    return dmap


def wrap_plot_fn_sample(input_ds, output_ds, plot_fn):
    samples = np.arange(input_ds.sample.size)

    def plot_selection(sample):
        inp, pred = (
            input_ds.isel(sample=sample),
            output_ds.isel(sample=sample),
        )
        return plot_fn(pred, inp)

    dmap = hv.DynamicMap(plot_selection, kdims=["sample"]).redim.values(sample=samples)
    return dmap


def flux_compare(eval_data: EvalData) -> hv.DynamicMap:
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

        return flux_plot

    # Create a DynamicMap for the heatmap with contour plot
    dmap_heatmap_contour = hv.DynamicMap(plot_selection, kdims=["time_idx", "shot"]).redim.values(time_idx=times, shot=shots)

    return dmap_heatmap_contour


def flux_and_diags(eval_data: EvalData) -> hv.DynamicMap:
    hv.extension("bokeh")

    times = eval_data.input_ds.time_idx.values
    shots = eval_data.input_ds.shot.values

    Bm_min = eval_data.input_ds["LY.Bm"].min().item()
    Bm_max = eval_data.input_ds["LY.Bm"].max().item()
    Ff_min = eval_data.input_ds["LY.Ff"].min().item()
    Ff_max = eval_data.input_ds["LY.Ff"].max().item()

    def plot_selection(time_idx, shot):
        inp, pred = (
            eval_data.input_ds.sel(time_idx=time_idx, shot=shot).squeeze().drop_vars("sample"),
            eval_data.output_ds.sel(time_idx=time_idx, shot=shot).squeeze().drop_vars("sample"),
        )

        Fx_relative_error = np.linalg.norm(pred["Fx"] - inp["LY.Fx"]) / np.linalg.norm(inp["LY.Fx"])

        flux_plot = plot_flux(
            {"Target": inp["LY.Fx"], f"Predicted (rel err: {Fx_relative_error:.3f})": pred["Fx"]},
            title=f"shot {shot} at time_idx {time_idx}",
        )

        Bm_relative_error = np.linalg.norm(pred["Bm"] - inp["LY.Bm"]) / np.linalg.norm(inp["LY.Bm"])

        Bm_plot = hv.Scatter(pred["Bm"].data, label=f"Predicted Bm (rel err: {Bm_relative_error:.3f})").opts(
            size=6, color="blue", marker="x"
        ) * hv.Scatter(inp["LY.Bm"].data, label="Target Bm").opts(size=6, ylim=(Bm_min, Bm_max), color="red", marker="o")

        Ff_relative_error = np.linalg.norm(pred["Ff"] - inp["LY.Ff"]) / np.linalg.norm(inp["LY.Ff"])

        Ff_plot = hv.Scatter(pred["Ff"].data, label=f"Predicted Ff (rel err: {Ff_relative_error:.3f})").opts(
            size=6, color="blue", marker="x"
        ) * hv.Scatter(inp["LY.Ff"].data, label="Target Ff").opts(size=6, ylim=(Ff_min, Ff_max), color="red", marker="o")

        return flux_plot + (Bm_plot + Ff_plot).cols(1)

    # Create a DynamicMap for the heatmap with contour plot
    dmap_heatmap_contour = hv.DynamicMap(plot_selection, kdims=["time_idx", "shot"]).redim.values(time_idx=times, shot=shots)

    return dmap_heatmap_contour


def control_points_pred_vs_targ(eval_data: EvalData):
    input_ds, output_ds = eval_data.input_ds, eval_data.output_ds

    output_ds = output_ds.drop_vars("sample")
    input_ds = input_ds.drop_vars("sample")
    xlim = (0.0, 1.2)
    ylim = (-1.0, 1.0)
    # aspect equal
    gui = (
        output_ds.hvplot.scatter(x="rc", y="zc", groupby="sample", label="pred", xlim=xlim, ylim=ylim, aspect="equal")
        * input_ds.hvplot.scatter(x="LY.SC.rc", y="LY.SC.zc", groupby="sample", label="targ", xlim=xlim, ylim=ylim, aspect="equal")
    ).opts(height=600)
    return gui


def control_point_error_hist(eval_data: EvalData):
    input_ds, output_ds = eval_data.input_ds, eval_data.output_ds

    r_error = np.abs(output_ds["rc"] - input_ds["LY.SC.rc"])
    z_error = np.abs(output_ds["zc"] - input_ds["LY.SC.zc"])

    dist = np.sqrt(r_error**2 + z_error**2)

    errors = dist.mean(dim="nc")

    _, ax = plt.subplots()

    ax.hist(errors.data, bins=100)
    ax.set_title("Distribution of Mean Control Point Errors (m)")
    ax.set_xlim(0.0, 0.02)
    return ax
