import os

import matplotlib.pyplot as plt
import xarray as xr

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18


def scope_dataset(ds: xr.Dataset, fig_dir: str) -> None:
    """Create plots for every shot in the dataset to visualize the data and ensure it looks fine."""
    for shot in ds["shot"].data:
        shot_ds = ds.sel(shot=shot)

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
        fig.patch.set_facecolor(BACKGROUND_COLOR)

        fig.suptitle(f"TCV Shot {shot}", fontsize=TITLE_FONTSIZE, color=TEXT_COLOR)

        # Ip and Wtot
        ax_ip = axes[0]
        # Put Ip on the left y axis and Wtot on the right y axis
        ax_ip.plot(shot_ds["time"], shot_ds["Ip_MA"], label="Ip [MA]", color="cyan")
        ax_ip.set_ylabel("Ip [MA]", fontsize=LABEL_FONTSIZE, color="cyan")
        ax_wtot = ax_ip.twinx()
        ax_wtot.plot(shot_ds["time"], shot_ds["Wtot_MJ"], label="Wtot [MJ]", color="red")
        ax_wtot.set_ylabel("Wtot [MJ]", fontsize=LABEL_FONTSIZE, color="red")
        ax_wtot.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)

        # powers
        ax_power = axes[1]
        ax_power.plot(shot_ds["time"], shot_ds["P_oh_MW"], label="P_oh [MW]", color="orange")
        ax_power.plot(shot_ds["time"], shot_ds["P_rad_MW"], label="P_rad [MW]", color="red")
        ax_power.plot(shot_ds["time"], shot_ds["P_NBI_MW"], label="P_NBI [MW]", color="cyan")
        ax_power.plot(shot_ds["time"], shot_ds["P_ECRH_MW"], label="P_ECRH [MW]", color="lime")
        ax_power.plot(shot_ds["time"], shot_ds["P_LH"] / 1e6, label="P_LH [MW]", color="white", linestyle="--")
        ax_power.set_ylabel("Power [MW]", fontsize=LABEL_FONTSIZE, color="white")
        ax_power.legend(fontsize=LEGEND_FONTSIZE, facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_COLOR, loc="upper left")

        # density
        ax_ne = axes[2]
        ax_ne.plot(shot_ds["time"], shot_ds["ne20_line_avg"], label="ne20_line_avg [m^-3]", color="white")
        ax_ne.set_ylabel("ne20_line_avg [m^-3]", fontsize=LABEL_FONTSIZE, color="white")

        # Shaping
        ax_shape = axes[3]
        ax_shape.plot(shot_ds["time"], shot_ds["a_minor"], label="a_minor", color="red")
        ax_shape.plot(shot_ds["time"], shot_ds["kappa"], label="kappa", color="yellow")
        ax_shape.plot(shot_ds["time"], shot_ds["delta_top"], label="delta_top", color="lime")
        ax_shape.plot(shot_ds["time"], shot_ds["delta_bottom"], label="delta_bottom", color="green")
        ax_shape.set_ylabel("Shaping", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.set_xlabel("Time [s]", fontsize=LABEL_FONTSIZE, color="white")
        ax_shape.legend(fontsize=LEGEND_FONTSIZE, facecolor=BACKGROUND_COLOR, edgecolor=BACKGROUND_COLOR, loc="upper left")

        for ax in axes:
            ax.set_facecolor(FACE_COLOR)
            ax.grid(True, color="gray", linestyle="--", linewidth=0.1)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors=TEXT_COLOR)
            try:
                for text in ax.get_legend().get_texts():
                    text.set_color(TEXT_COLOR)
            except AttributeError:
                pass

        fig.tight_layout()
        fig.savefig(f"{fig_dir}/{shot}.png")
        plt.close(fig)


if __name__ == "__main__":
    ds_path = "/usr/local/mfe/ml_data_dump/studies/transport_predictor/tcv/tcv_200_processed.zarr"
    fig_dir = "/usr/local/mfe/ml_data_dump/studies/transport_predictor/tcv/tcv_200"
    os.makedirs(fig_dir, exist_ok=True)
    ds = xr.open_zarr(ds_path, consolidated=True)
    scope_dataset(ds, fig_dir)
