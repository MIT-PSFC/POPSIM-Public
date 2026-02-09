import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from popsim.math_utils import padded_relative_error


def eval_relative_errors(eval_data):
    input_ds, output_ds = eval_data.input_ds.transpose("sample", ...), eval_data.output_ds.transpose("sample", ...)

    vec_padded_relative_error = jax.vmap(padded_relative_error, in_axes=(0, 0))

    errors = {
        "Fx": vec_padded_relative_error(output_ds.Fx.data, input_ds["LY.Fx"].data),
        "Ff": vec_padded_relative_error(output_ds.Ff.data, input_ds["LY.Ff"].data),
        "Bm": vec_padded_relative_error(output_ds.Bm.data, input_ds["LY.Bm"].data),
    }

    return errors


def relative_error_medians(eval_data):
    errors = eval_relative_errors(eval_data)
    return {k: jnp.median(v) for k, v in errors.items()}


def padded_relative_error_hists(eval_data, quantiles=(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)):
    errors = eval_relative_errors(eval_data)
    n_shots = np.unique(eval_data.input_ds.shot).size
    n_equils = np.unique(eval_data.input_ds.sample).size
    f, axs = plt.subplots(1, 3, figsize=(15, 5))

    f.suptitle(f"Relative errors for {n_shots} shots and {n_equils} equilibria")

    # Define a list of colors corresponding to the quantiles.
    colors = ["red", "green", "blue", "orange", "purple", "brown", "pink"]

    for ax, (name, err) in zip(axs, errors.items(), strict=True):
        ax.hist(err, bins=50, density=True, alpha=0.7)

        # Plot a vertical line for each quantile with a unique color and label.
        for q, color in zip(quantiles, colors, strict=True):
            q_val = jnp.quantile(err, q)
            ax.axvline(q_val, color=color, linestyle="--", label=f"Quantile {q}")

        ax.set_title(name)
        ax.set_xlabel("Relative error")
        ax.set_ylabel("Density")
        ax.legend()  # Add legend to the subplot

    return f
