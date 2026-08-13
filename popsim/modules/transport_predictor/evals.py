import matplotlib.pyplot as plt

from popsim.ml import EvalData

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def plot_profiles(eval_data: EvalData, title_suffix=""):
    """Plots predicted vs target profiles from the transport predictor module.

    Dark-dashed is target, bright-solid is predicted.

    Args:
        eval_data: EvalData object containing input and output datasets
        title_suffix: Optional suffix to add to plot titles

    Returns:
        dict: Dictionary mapping shot numbers to (figure, axes) tuples
    """
    rho_colors = {
        0: ("firebrick", "red"),
        0.4: ("darkorange", "orange"),
        0.6: ("khaki", "yellow"),
        0.8: ("limegreen", "lime"),
        0.9: ("darkturquoise", "cyan"),
        1.0: ("blueviolet", "violet"),
    }

    # Get the combined dataset
    # Output variables are already prefixed with "output." or "state."
    # Input and output profile variables share the indexed "rho" dimension.
    ds = eval_data.input_ds.copy()
    for var in eval_data.output_ds.data_vars:
        ds[var] = eval_data.output_ds[var]

    fig_size = (12, 12)
    figures = {}

    for shot in ds.shot.values:
        shot_data = ds.where(ds.shot == shot, drop=True).squeeze(drop=True)
        fig, axes = plt.subplots(2, 1, figsize=fig_size)
        fig.suptitle(f"Profiles for TCV Shot {shot}, {title_suffix}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        # Plot the electron density profile
        for rho, (target_color, pred_color) in rho_colors.items():
            ne_target = shot_data["ne20_rho"].sel(rho=rho, method="nearest")
            ne_predicted = shot_data["output.profile_predictor_output.ne"].sel(rho=rho, method="nearest")
            axes[0].plot(shot_data.time, ne_target, label=f"ne {rho} Target", color=target_color, linestyle="--")
            axes[0].plot(shot_data.time, ne_predicted, label=f"ne {rho} Predicted", color=pred_color)

        axes[0].set_title("Electron Density Profiles", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        axes[0].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        axes[0].set_ylabel("Electron Density [10^20 m^-3]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

        # Plot the electron temperature profile
        for rho, (target_color, pred_color) in rho_colors.items():
            te_target = shot_data["Te_keV_rho"].sel(rho=rho, method="nearest")
            te_predicted = shot_data["output.profile_predictor_output.te"].sel(rho=rho, method="nearest")
            axes[1].plot(shot_data.time, te_predicted, label=f"Te {rho} Predicted", color=pred_color)
            axes[1].plot(shot_data.time, te_target, label=f"Te {rho} Target", color=target_color, linestyle="--")

        axes[1].set_title("Electron Temperature Profiles", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        axes[1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        axes[1].set_ylabel("Electron Temperature [keV]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

        for ax in axes:
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
            ax.set_facecolor(FACE_COLOR)
            # No legend because too many lines
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1.1)

        fig.tight_layout()
        figures[shot] = (fig, axes)

    return figures


def plot_power_balance(eval_data: EvalData):
    """Plots related to the power balance module.

    Args:
        eval_data: EvalData object containing input and output datasets

    Returns:
        dict: Dictionary mapping shot numbers to (figure, axes) tuples
    """
    # Get the combined dataset
    ds = eval_data.input_ds.copy()
    for var in eval_data.output_ds.data_vars:
        ds[f"output.{var}"] = eval_data.output_ds[var]

    # Also add state variables if available
    if hasattr(eval_data, "state_ds") and eval_data.state_ds is not None:
        for var in eval_data.state_ds.data_vars:
            ds[f"state.{var}"] = eval_data.state_ds[var]

    fig_size = (12, 8)
    figures = {}

    for shot in ds.shot.values:
        shot_data = ds.where(ds.shot == shot, drop=True).squeeze(drop=True)
        fig, axes = plt.subplots(2, 1, figsize=fig_size)
        fig.suptitle(f"Power Balance for TCV Shot {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        # Plot the predicted stored energy vs the actual stored energy
        if "Wtot_MJ" in shot_data and "state.power_balance_state.stored_energy" in shot_data:
            stored_energy_target = shot_data["Wtot_MJ"]
            stored_energy_predicted = shot_data["state.power_balance_state.stored_energy"]
            axes[0].plot(shot_data.time, stored_energy_target, label="Target", color="purple", linewidth=2, linestyle="--")
            axes[0].plot(shot_data.time, stored_energy_predicted, label="Predicted", color="pink", linewidth=2)
        axes[0].set_title("Stored Energy", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        axes[0].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        axes[0].set_ylabel("Stored Energy [MJ]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

        # Plot the predicted power balance vs the actual power balance
        if "p_oh_MW" in shot_data and "output.p_oh_output.p_oh_MW_predicted" in shot_data:
            p_oh_target = shot_data["p_oh_MW"]
            p_oh_predicted = shot_data["output.p_oh_output.p_oh_MW_predicted"]
            axes[1].plot(shot_data.time, p_oh_target, label="P_oh Target", color="blue", linewidth=2, linestyle="--")
            axes[1].plot(shot_data.time, p_oh_predicted, label="P_oh Predicted", color="cyan", linewidth=2)

        if "p_rad_MW" in shot_data and "output.p_rad_output.p_rad_MW_predicted" in shot_data:
            p_rad_target = shot_data["p_rad_MW"]
            p_rad_predicted = shot_data["output.p_rad_output.p_rad_MW_predicted"]
            axes[1].plot(shot_data.time, p_rad_target, label="P_rad Target", color="red", linewidth=2, linestyle="--")
            axes[1].plot(shot_data.time, p_rad_predicted, label="P_rad Predicted", color="orange", linewidth=2)

        axes[1].set_title("Power Sources", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)
        axes[1].set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)
        axes[1].set_ylabel("Power [MW]", fontsize=LABEL_FONTSIZE, color=TEXT_COLOR)

        for ax in axes:
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            ax.legend(facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_COLOR, fontsize=LEGEND_FONTSIZE, loc="upper left")
            ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
            ax.set_facecolor(FACE_COLOR)

            # Make text of legend match text color
            try:
                for text in ax.get_legend().get_texts():
                    text.set_color(TEXT_COLOR)
            except AttributeError:
                pass

        fig.tight_layout()
        figures[shot] = (fig, axes)

    return figures
