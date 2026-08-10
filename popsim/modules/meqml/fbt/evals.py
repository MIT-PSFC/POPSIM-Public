import numpy as np
import xarray as xr

from popsim.ml.eval import EvalData
from popsim.modules.meqml.eval_utils import wrap_plot_fn_shot


def eval_ia_prediction_errors(eval_data: EvalData):
    Ia_pred = eval_data.output_ds["Ia"]
    Ia_targ = eval_data.input_ds["LY.Ia"]

    Ia_error = np.abs(Ia_pred - Ia_targ)
    return Ia_error


def eval_ia_prediction_error_quantiles(eval_data: EvalData, quantiles: tuple = (0.5, 0.9, 0.99, 0.999), min_shot_number=None):
    Ia_error = eval_ia_prediction_errors(eval_data)

    Ia_error = Ia_error.mean("active_coils")

    if min_shot_number is not None:
        Ia_error = Ia_error.where(Ia_error["shot"] >= min_shot_number)

    Ia_error_quantiles = Ia_error.quantile(quantiles, "sample")

    quantile_dict = {f"quantile{q}": Ia_error_quantiles.sel(quantile=q).data for q in quantiles}
    return quantile_dict


def eval_ia_pred_targ(eval_data):
    import holoviews as hv

    input_ds, output_ds = eval_data.input_ds, eval_data.output_ds

    def plot_ia_pred_targ(pred, inp):
        in_ia = inp["LY.Ia"]
        out_ia = pred["Ia"]

        def make_plot_for_act(act):
            in_data = in_ia.sel(active_coils=act)
            out_data = out_ia.sel(active_coils=act)

            min_y = min(in_data.min().item(), out_data.min().item())
            max_y = max(in_data.max().item(), out_data.max().item())

            plot = (out_data.hvplot.line(x="time", label="Predicted Ia") * in_data.hvplot.line(x="time", label="Target Ia")).opts(
                ylim=(min_y, max_y), width=400, height=200, title=f"Active coil {act}"
            )

            return plot

        act_plots = [make_plot_for_act(act) for act in out_ia.active_coils.data]

        return hv.Layout(act_plots).cols(3).opts(shared_axes=False)

    return wrap_plot_fn_shot(input_ds, output_ds, plot_ia_pred_targ)


def eval_sorted_gui(eval_data):
    input_ds, output_ds = eval_data.input_ds, eval_data.output_ds
    mean_errors = np.abs(input_ds["LY.Ia"] - output_ds["Ia"]).mean("active_coils")
    ds = xr.merge([input_ds, output_ds, mean_errors.rename("mean_error")], compat="no_conflicts", join="exact")
    ds = ds[["LY.Ia", "Ia", "mean_error", "LY.SC.rc", "LY.SC.zc"]].sortby("mean_error").drop_vars("sample").isel(sample=slice(0, None, 10))
    out = ds.hvplot.scatter(x="LY.SC.rc", y="LY.SC.zc", groupby="sample") + (
        ds.hvplot.scatter(x="active_coils", y="LY.Ia", groupby="sample") * ds.hvplot.scatter(x="active_coils", y="Ia", groupby="sample")
    ).opts(ylim=(-3500, 3500))
    return out


def mean_error_hist(eval_data):
    import matplotlib.pyplot as plt

    ia_pred_errors = eval_ia_prediction_errors(eval_data)
    # Load and preprocess
    Ia_pred_errors = ia_pred_errors.rename("Ia Pred Errors").drop_vars("sample")
    mean_Ia_pred_errors = Ia_pred_errors.mean("active_coils")  # Mean across active coils

    # Plot settings
    f, ax = plt.subplots(figsize=(5, 4))

    quantiles = [0.5, 0.9, 0.95, 0.99, 0.999]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]  # Different styles

    # Data
    data = mean_Ia_pred_errors.values.flatten()

    # Compute quantiles
    abs_data = np.abs(data)
    quantile_values = np.quantile(abs_data, quantiles)
    max_quantile = quantile_values[-1]

    # Plot histogram (count, not density)
    ax.hist(data, bins=50, density=False, alpha=0.7, color="gray", edgecolor="black", linewidth=0.5)

    # Plot vertical lines and build legend
    handles = []
    labels = []
    for qv, q, ls in zip(quantile_values, quantiles, linestyles, strict=True):
        h = ax.axvline(qv, linestyle=ls, color="red", linewidth=1.5, label=f"{q:.3f}: {qv:.1f} A")
        handles.append(h)
        labels.append(f"{q:.3f}: {qv:.1f} A")

    n_samples = data.size

    ax.set_title(f"Mean Prediction Error Across Active Coils ({n_samples} samples)")
    ax.set_xlabel("Prediction Error (A)")
    ax.set_ylabel("Number of Samples")
    ax.set_xlim(0, max_quantile * 1.1)
    ax.legend(handles, labels, fontsize=8, loc="upper right", framealpha=0.8)

    f.tight_layout()
    return f
