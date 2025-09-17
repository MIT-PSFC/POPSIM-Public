import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from popsim.ml import EvalData


def violin_shapes_in_data(eval_data: EvalData, rho_downsample: int = 1):
    variables_to_plot = ["Te_shape", "ne_shape"]
    # Create subplots
    _f, axes = plt.subplots(nrows=len(variables_to_plot), figsize=(10, 6 * len(variables_to_plot)), sharex=True)

    if len(variables_to_plot) == 1:  # Handle single subplot case
        axes = [axes]

    for var_name, ax in zip(variables_to_plot, axes, strict=False):
        # Extract the variable
        data_array = eval_data.input_ds[var_name]

        # Prepare the data for seaborn
        rho_values = data_array["rho"].values[::rho_downsample]
        samples = [data_array.sel(rho=rho).values for rho in rho_values]

        # Plot the violin plot
        sns.violinplot(data=samples, scale="width", ax=ax)

        # Customize each subplot
        ax.set_title(f"Violin Plot of {var_name}")
        ax.set_xticks(range(len(rho_values)))
        ax.set_xticklabels([f"{rho:.2f}" for rho in rho_values])
        ax.set_xlabel("Rho")
        ax.set_ylabel("Values")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def compute_integrated_error(eval_data: EvalData, q=None, return_distributions=False):
    if q is None:
        q = [0.01, 0.1, 0.5, 0.9, 0.99]
    output_ds = eval_data.output_ds
    input_ds = eval_data.input_ds

    output_ds["integrated_ne_error"] = xr.apply_ufunc(jnp.abs, output_ds["ne"] - input_ds["ne20_rho"]).integrate("rho")
    output_ds["integrated_te_error"] = xr.apply_ufunc(jnp.abs, output_ds["te"] - input_ds["Te_keV_rho"]).integrate("rho")
    output_ds["integrated_ne_percent_error"] = 100.0 * (output_ds["integrated_ne_error"] / input_ds["ne20_rho"].integrate("rho"))
    output_ds["integrated_te_percent_error"] = 100.0 * (output_ds["integrated_te_error"] / input_ds["Te_keV_rho"].integrate("rho"))

    def quantile_dict(da, q):
        quants = da.quantile(q, dim="sample")
        return dict(zip(quants["quantile"].values, quants.values, strict=False))

    out_dict = {}

    # Compute quantiles.
    out_dict["integrated_ne_error_quantiles"] = quantile_dict(output_ds["integrated_ne_error"], q)
    out_dict["integrated_te_error_quantiles"] = quantile_dict(output_ds["integrated_te_error"], q)
    out_dict["integrated_ne_percent_error_quantiles"] = quantile_dict(output_ds["integrated_ne_percent_error"], q)
    out_dict["integrated_te_percent_error_quantiles"] = quantile_dict(output_ds["integrated_te_percent_error"], q)

    if return_distributions:
        out_dict["integrated_ne_error"] = output_ds["integrated_ne_error"]
        out_dict["integrated_te_error"] = output_ds["integrated_te_error"]
        out_dict["integrated_ne_percent_error"] = output_ds["integrated_ne_percent_error"]
        out_dict["integrated_te_percent_error"] = output_ds["integrated_te_percent_error"]

    return out_dict


def compare_profiles_for_episode(eval_data: EvalData, episode_idx=0):
    input_ds = eval_data.input_ds.unstack("sample")
    output_ds = eval_data.output_ds.unstack("sample")

    f, axs = plt.subplots(2, 1, figsize=(15, 10))

    rhos_plot = [0.0, 0.3, 0.6, 0.9, 1.0]

    episode_dim = "simulation"

    input_ds = input_ds.isel({episode_dim: episode_idx})
    output_ds = output_ds.isel({episode_dim: episode_idx})
    targ_ne = input_ds["ne20_rho"]
    targ_te = input_ds["Te_keV_rho"]
    pred_ne = output_ds["ne"]
    pred_te = output_ds["te"]

    for rho in rhos_plot:
        # Ne Plot
        axs[0].plot(pred_ne["time"], pred_ne.sel(rho=rho, method="nearest").values, color="b", linewidth=2, label="Predicted")
        axs[0].plot(targ_ne["time"], targ_ne.sel(rho=rho, method="nearest").values, color="k", marker="o", linestyle="-", label="Target")
        axs[0].annotate(
            rf"$\rho={rho}$",
            xy=(pred_ne["time"][-1], pred_ne.sel(rho=rho, method="nearest").values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=12,
            ha="left",
            va="center",
        )
        axs[0].set_title("Ne")
        axs[0].legend()

        # Te Plot
        axs[1].plot(pred_te["time"], pred_te.sel(rho=rho, method="nearest").values, color="r", linewidth=2, label="Predicted")
        axs[1].plot(targ_te["time"], targ_te.sel(rho=rho, method="nearest").values, color="k", marker="o", linestyle="-", label="Target")
        axs[1].annotate(
            rf"$\rho={rho}$",
            xy=(pred_te["time"][-1], pred_te.sel(rho=rho, method="nearest").values[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=12,
            ha="left",
            va="center",
        )
        axs[1].set_title("Te")
        axs[1].legend()

    return f, axs


def make_histograms(eval_data: EvalData, bins=100):
    integrated_error_data = compute_integrated_error(eval_data, return_distributions=True)

    ne_error = integrated_error_data["integrated_ne_error"].values
    ne_error_percent = integrated_error_data["integrated_ne_percent_error"].values
    te_error = integrated_error_data["integrated_te_error"].values
    te_error_percent = integrated_error_data["integrated_te_percent_error"].values

    f, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0][0].hist(ne_error, bins=bins, alpha=0.5, label="ne error")
    axs[0][0].set_title("Integrated ne error")
    axs[0][0].set_xlabel("Error")
    axs[0][0].set_ylabel("Count")
    axs[0][0].legend()

    axs[0][1].hist(ne_error_percent, bins=bins, alpha=0.5, label="ne error percent")
    axs[0][1].set_title("Integrated ne error percent")
    axs[0][1].set_xlabel("Error (%)")
    axs[0][1].set_ylabel("Count")

    axs[1][0].hist(te_error, bins=bins, alpha=0.5, label="te error")
    axs[1][0].set_title("Integrated Te error")
    axs[1][0].set_xlabel("Error")
    axs[1][0].set_ylabel("Count")
    axs[1][0].legend()

    axs[1][1].hist(te_error_percent, bins=bins, alpha=0.5, label="te error percent")
    axs[1][1].set_title("Integrated Te error percent")
    axs[1][1].set_xlabel("Error (%)")
    axs[1][1].set_ylabel("Count")
    axs[1][1].legend()
    return f, axs


def make_quantile_examples(eval_data: EvalData, quantiles=None):
    if quantiles is None:
        quantiles = [0.01, 0.1, 0.5, 0.9, 0.99]
    integrated_error_data = compute_integrated_error(eval_data, return_distributions=True)

    f, axs = plt.subplots(len(quantiles), 2, figsize=(10, 4 * len(quantiles)))

    def make_row_for_quantile(quantile, ne_ax, te_ax):
        quantile_val = integrated_error_data["integrated_ne_error"].quantile(quantile, method="nearest")
        sample_at_quantile = (
            integrated_error_data["integrated_ne_error"]
            .where(integrated_error_data["integrated_ne_error"] == quantile_val)
            .dropna("sample")
        )
        sample_inp = eval_data.input_ds.sel(sample=sample_at_quantile.sample.values)
        sample_out = eval_data.output_ds.sel(sample=sample_at_quantile.sample.values)
        ne_ax.scatter(sample_inp["rho"], sample_inp["ne20_rho"].values.squeeze(), label="Target")
        ne_ax.plot(sample_out["rho"], sample_out["ne"].values.squeeze(), label="Predicted")
        ne_ax.set_title(f"Ne Error at Quantile {quantile}")
        ne_ax.set_ylim(0, 1.2 * sample_inp["ne20_rho"].max())
        ne_ax.legend()

        quantile_val = integrated_error_data["integrated_te_error"].quantile(quantile, method="nearest")
        sample_at_quantile = (
            integrated_error_data["integrated_te_error"]
            .where(integrated_error_data["integrated_te_error"] == quantile_val)
            .dropna("sample")
        )
        sample_inp = eval_data.input_ds.sel(sample=sample_at_quantile.sample.values)
        sample_out = eval_data.output_ds.sel(sample=sample_at_quantile.sample.values)
        te_ax.scatter(sample_inp["rho"], sample_inp["Te_keV_rho"].values.squeeze(), label="Target")
        te_ax.plot(sample_out["rho"], sample_out["te"].values.squeeze(), label="Predicted")
        te_ax.set_title(f"Te Error at Quantile {quantile}")
        te_ax.set_ylim(0, 1.2 * sample_inp["Te_keV_rho"].max())
        te_ax.legend()

    for i, quantile in enumerate(quantiles):
        make_row_for_quantile(quantile, axs[i][0], axs[i][1])
    return f, axs


def plot_shapes(eval_data: EvalData):
    model = eval_data.model

    # Plot all of the Te shapes.
    te_shapes = model.te_shapes
    ne_shapes = model.ne_shapes

    f, axs = plt.subplots(1, 2, figsize=(10, 5))
    rho = np.linspace(0, 1, 100)

    # Plot te_shapes
    for i, shape in enumerate(te_shapes):
        shape_eval = shape(rho)
        axs[0].plot(rho, shape_eval, label=f"Te Shape {i}")

    axs[0].legend()
    axs[0].set_title("Te Shapes")
    axs[0].set_xlabel("rho")

    # Plot ne_shapes
    for i, shape in enumerate(ne_shapes):
        shape_eval = shape(rho)
        axs[1].plot(rho, shape_eval, label=f"Ne Shape {i}")

    axs[1].legend()
    axs[1].set_title("Ne Shapes")
    axs[1].set_xlabel("rho")
    return f, axs


def plot_ne_te_weights(eval_data: EvalData):
    if "debug_info.ne_coeffs" not in eval_data.output_ds or "debug_info.te_coeffs" not in eval_data.output_ds:
        return None, None
    ne_weights = eval_data.output_ds["debug_info.ne_coeffs"].transpose("sample", ...).values
    te_weights = eval_data.output_ds["debug_info.te_coeffs"].transpose("sample", ...).values

    n_ne_weights = ne_weights.shape[1]
    n_te_weights = te_weights.shape[1]
    f, axs = plt.subplots(1, n_ne_weights, figsize=(5 * n_ne_weights, 3))
    for i in range(n_ne_weights):
        axs[i].hist(ne_weights[:, i])
        axs[i].set_title(f"Ne Weight {i}")

    f, axs = plt.subplots(1, n_te_weights, figsize=(5 * n_te_weights, 3))
    for i in range(n_te_weights):
        axs[i].hist(te_weights[:, i])
        axs[i].set_title(f"Te Weight {i}")

    return f, axs
