import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from popsim.math_utils import padded_relative_error
from popsim.ml.eval import EvalData


def eval_padded_relative_errors(eval_data: EvalData, pad: float = 1.0):
    input_ds, output_ds = eval_data.input_ds, eval_data.output_ds

    errors = {
        "Ip": padded_relative_error(output_ds.Ip.data, input_ds["LY.Ip"].data, pad=pad),
        "rIp": padded_relative_error(output_ds.rIp.data, input_ds["LY.rIp"].data, pad=pad),
        "zIp": padded_relative_error(output_ds.zIp.data, input_ds["LY.zIp"].data, pad=pad),
    }

    return errors


def eval_relative_error_means(eval_data: EvalData):
    errors = eval_padded_relative_errors(eval_data)
    return {f"{k}_mean": float(jnp.mean(v)) for k, v in errors.items()}


def padded_relative_error_hist(eval_data: EvalData, quantiles=(0.01, 0.1, 0.5, 0.9, 0.99)):
    errors = eval_padded_relative_errors(eval_data)
    n_shots = np.unique(eval_data.input_ds.shot).size if "shot" in eval_data.input_ds.dims else 1
    n_samples = eval_data.input_ds.sample.size

    f, axs = plt.subplots(1, 3, figsize=(15, 5))
    f.suptitle(f"Relative errors for {n_shots} shots and {n_samples} samples")

    colors = ["red", "green", "blue", "orange", "purple"]

    for ax, (name, err) in zip(axs, errors.items(), strict=True):
        ax.hist(err, bins=50, density=True, alpha=0.7, color="gray", edgecolor="black", linewidth=0.5)

        for q, color in zip(quantiles, colors, strict=True):
            q_val = jnp.quantile(err, q)
            ax.axvline(q_val, color=color, linestyle="--", linewidth=1.5, label=f"{q:.3f}: {q_val:.3f}")

        ax.set_title(f"{name} Relative Error")
        ax.set_xlabel("Relative error")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)

    plt.tight_layout()
    return f
