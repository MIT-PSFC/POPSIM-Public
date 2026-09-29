"""Flag anomalous samples in a tensorized dataset with a self-supervised model.

Curating a large dataset by hand is impractical, and broken or noisy data corrupts training and the reported
validation statistics. This script trains a model on the dataset itself and flags the samples it fits worst,
so a human expert only has to review the flagged episodes and times.

Two modes:
    time_indep (default): an MLP autoencoder over every (episode, time) sample, score = reconstruction error.
        Catches samples that are off the manifold of the rest of the data.
    time_dep: a causal transformer over segments of consecutive time steps that predicts every variable at
        step t + 1 from all variables up to step t, score = one-step-ahead prediction error per time step.
        Catches transient events and sequences that are individually plausible but inconsistent in time.

Model sizes are derived from the covariance spectrum of the normalized data, see choose_n_latent.
Scores are converted to a robust z-score of the log score (median and MAD), samples above z_threshold are flagged.

Example:
    popsim-autocheck path/to/dataset.zarr out_dir --mode time_dep --segment_length 40 --segment_overlap 20
"""

import functools
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import equinox as eqx
import fire
import jax.numpy as jnp
import loguru
import numpy as np
import optax
import pandas as pd
import tabulate
import xarray as xr
from jaxtyping import Array

from popsim.ml import DataLoader, Trainer, TrainRunBuilder, make_time_dep_dataloader, make_time_indep_dataloader
from popsim.ml.dataloading import make_dataloaders
from popsim.ml.envs import ModuleEvalEnvInput
from popsim.ml.eval import EvalData
from popsim.ml.launch import launch_train
from popsim.ml.split_utils import fracs_to_lengths, split_dataset_by_fracs
from popsim.modules.causal_transformer import CausalTransformerPredictor
from popsim.modules.mlp_autoencoder import MLPAutoEncoder

MODE_TIME_INDEP = "time_indep"
MODE_TIME_DEP = "time_dep"
MODES = (MODE_TIME_INDEP, MODE_TIME_DEP)

LOSS_VAR = "loss"
VAR_ERROR_VAR = "var_error"
STEP_ERROR_VAR = "step_error"
STEP_VAR_ERROR_VAR = "step_var_error"
VARIABLE_DIM = "variable"
ERROR_COLUMN_PREFIX = "error."

SCORES_FILE = "autocheck_scores.nc"
FLAGGED_FILE = "flagged.csv"
EPISODE_SUMMARY_FILE = "episode_summary.csv"
CHECKPOINT_SUBDIR = "checkpoints"
TRAIN_RUN_BUILDER = "popsim.data.autocheck.AutocheckTrainRunBuilder"

MAD_TO_STD = 1.4826
SCORE_FLOOR = 1e-12
MAX_REPORT_ROWS = 30


class AutocheckScorer(eqx.Module):
    """Wrap a model so its output is the per-sample score the trainer minimizes and the report consumes.

    Output per sample: loss (scalar), var_error (variable,), and for a time-dependent model also
    step_error (time,) and step_var_error (time, variable), NaN at steps without a valid prediction.
    """

    model: MLPAutoEncoder | CausalTransformerPredictor

    def __call__(self, inputs: xr.Dataset | ModuleEvalEnvInput) -> dict[str, xr.Variable]:
        if isinstance(inputs, ModuleEvalEnvInput):
            sq_err, mask_valid = self.model.squared_normalized_error(inputs.inputs, inputs.time)
            return self._score(sq_err, mask_valid, with_steps=True)
        sq_err = self.model.squared_normalized_error(inputs)
        return self._score(sq_err[None, :], jnp.ones(1, dtype=bool), with_steps=False)

    def _score(self, sq_err: Array, mask_valid: Array, with_steps: bool) -> dict[str, xr.Variable]:
        flattener = self.model.flattener
        weights = mask_valid.astype(sq_err.dtype)[:, None]
        n_valid = jnp.maximum(mask_valid.sum(), 1)
        loss = jnp.sum(sq_err * weights) / (n_valid * flattener.n_flat)
        step_var_error = jnp.stack([sq_err[:, start:stop].mean(axis=1) for start, stop in flattener.flat_slices.values()], axis=1)
        var_error = jnp.sum(step_var_error * weights, axis=0) / n_valid
        out = {LOSS_VAR: xr.Variable((), loss), VAR_ERROR_VAR: xr.Variable((VARIABLE_DIM,), var_error)}
        if with_steps:
            time_dim = self.model.time_dim
            step_error = jnp.where(mask_valid, sq_err.mean(axis=1), jnp.nan)
            out[STEP_ERROR_VAR] = xr.Variable((time_dim,), step_error)
            out[STEP_VAR_ERROR_VAR] = xr.Variable((time_dim, VARIABLE_DIM), jnp.where(weights > 0, step_var_error, jnp.nan))
        return out


def episode_and_time_dims(ds: xr.Dataset, episode_coord: str, time_coord: str) -> tuple[str, str]:
    """The episode dimension and the time dimension of a multi-episode dataset."""
    episode_dim = ds[episode_coord].dims[0]
    time_dims = [d for d in ds[time_coord].dims if d != episode_dim]
    if len(time_dims) != 1:
        raise ValueError(f"Expected one time dimension besides {episode_dim} on {time_coord}, got {time_dims}.")
    return episode_dim, time_dims[0]


def open_autocheck_dataset(dataset_path: str, time_coord: str) -> xr.Dataset:
    """Open a tensorized zarr store or a netCDF file with the same layout, with time as a coordinate."""
    if str(dataset_path).endswith(".zarr"):
        ds = xr.open_zarr(dataset_path)
    else:
        ds = xr.open_dataset(dataset_path)
    if time_coord in ds.data_vars:
        ds = ds.set_coords(time_coord)
    if time_coord not in ds.coords:
        raise ValueError(f"Time coordinate {time_coord} not found in {dataset_path}.")
    return ds


def select_variables(ds: xr.Dataset, variables: Sequence[str] | None, episode_coord: str, time_coord: str, mode: str) -> list[str]:
    """The variables to model: every numeric time-varying data var unless given explicitly.

    Static (no time dimension) variables are only allowed in time-independent mode.
    """
    episode_dim, time_dim = episode_and_time_dims(ds, episode_coord, time_coord)
    if variables is None:
        variables = [
            name
            for name, da in ds.data_vars.items()
            if episode_dim in da.dims and time_dim in da.dims and np.issubdtype(da.dtype, np.number)
        ]
        if not variables:
            raise ValueError("No numeric variables with both the episode and time dimensions found.")
        return variables

    variables = list(variables)
    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Variables {missing} not found in the dataset.")
    static = [name for name in variables if time_dim not in ds[name].dims]
    if static and mode == MODE_TIME_DEP:
        raise ValueError(f"Variables {static} have no {time_dim} dimension, which time-dependent mode requires.")
    return variables


class AutocheckTrainRunBuilder(TrainRunBuilder):
    @staticmethod
    def get_dataloaders(config: dict) -> tuple[xr.Dataset, DataLoader, DataLoader, DataLoader]:
        """Train and validation loaders split by episode, plus an unshuffled loader over every sample for scoring."""
        mode = config["mode"]
        episode_coord, time_coord = config["episode_coord"], config["time_coord"]
        ds = open_autocheck_dataset(config["dataset_path"], time_coord)
        variables = select_variables(ds, config["variables"], episode_coord, time_coord, mode)
        ds = ds[variables]
        for name in variables:
            if not np.issubdtype(ds[name].dtype, np.floating):
                ds[name] = ds[name].astype(float)
        ds = ds.load()
        loguru.logger.info(f"Loaded {len(variables)} variables, {ds.nbytes / 1e9:.2f} GB in memory.")

        episode_dim, _ = episode_and_time_dims(ds, episode_coord, time_coord)
        split_fracs = (1.0 - config["val_frac"], config["val_frac"])
        n_episodes = ds.sizes[episode_dim]
        if min(fracs_to_lengths(n_episodes, split_fracs)) == 0:
            raise ValueError(f"val_frac={config['val_frac']} leaves an empty split for {n_episodes} episodes, raise it.")
        train_ds, val_ds = split_dataset_by_fracs(ds, split_fracs, episode_dim, config["seed"])

        loader_kwargs = dict(
            time_coord=time_coord,
            episode_coord=episode_coord,
            input_vars=variables,
            target_vars=variables,
            convert_xr_to_jnp=False,
            batch_size=config["batch_size"],
        )
        if mode == MODE_TIME_DEP:
            segment_kwargs = dict(
                state_init_vars=[],
                segment_length=config["segment_length"],
                segment_overlap=config["segment_overlap"],
                nan_handling="drop_segment",
            )
            train_dl, val_dl = make_dataloaders([train_ds, val_ds], shuffle=[True, False], **loader_kwargs, **segment_kwargs)
            score_dl = make_time_dep_dataloader(ds, shuffle=False, **loader_kwargs, **segment_kwargs)
        else:
            train_dl, val_dl = make_dataloaders([train_ds, val_ds], shuffle=[True, False], **loader_kwargs)
            score_dl = make_time_indep_dataloader(ds, shuffle=False, **loader_kwargs)
        return ds, train_dl, val_dl, score_dl

    @staticmethod
    def model_init(train_dl: DataLoader, model_init_config: dict) -> AutocheckScorer:
        config = dict(model_init_config)
        mode = config.pop("mode")
        if mode == MODE_TIME_DEP:
            model = CausalTransformerPredictor.init(train_dl, **config)
        else:
            model = MLPAutoEncoder.init(train_dl, **config)
        return AutocheckScorer(model=model)

    @staticmethod
    def get_loss_fn(config: dict):
        # The scorer already computed the per-sample loss, targets equal the inputs and are unused.
        def loss_fn(pred, targ):
            return pred[LOSS_VAR].data

        return loss_fn

    @staticmethod
    def get_optimizer(config: dict) -> optax.GradientTransformation:
        return optax.adam(learning_rate=config["learning_rate"])

    @staticmethod
    def get_trainable_getter(config: dict):
        # Normalization statistics inside the model stay frozen.
        return lambda scorer: scorer.model.trainable()

    @staticmethod
    def get_test_eval_suite(config: dict):
        return {"scores": functools.partial(eval_autocheck_scores, time_coord=config["time_coord"])}


def eval_autocheck_scores(eval_data: EvalData, time_coord: str) -> xr.Dataset:
    """Per-sample scores of the scorer on the whole loader, with variable names and the time coordinate attached."""
    variables = list(eval_data.model.model.flattener.variables)
    scores = eval_data.output_ds.assign_coords({VARIABLE_DIM: variables})
    scores = scores.assign_coords({time_coord: eval_data.dataloader.ds[time_coord]})
    return scores


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """z-score with the median and the scaled median absolute deviation, so outliers do not inflate the scale."""
    median = np.nanmedian(x)
    mad = MAD_TO_STD * np.nanmedian(np.abs(x - median))
    mad = max(mad, np.finfo(float).eps)
    return (x - median) / mad


def _time_indep_score_table(scores: xr.Dataset, episode_coord: str, time_coord: str, time_dim: str) -> pd.DataFrame:
    """One row per (episode, time) sample."""
    table = pd.DataFrame({episode_coord: scores[episode_coord].values})
    if time_dim != time_coord:
        table[time_dim] = scores[time_dim].values
    table[time_coord] = scores[time_coord].values
    table["score"] = scores[LOSS_VAR].values
    for i, name in enumerate(scores[VARIABLE_DIM].values):
        table[f"{ERROR_COLUMN_PREFIX}{name}"] = scores[VAR_ERROR_VAR].values[:, i]
    return table


def _time_dep_score_table(scores: xr.Dataset, episode_coord: str, time_coord: str) -> pd.DataFrame:
    """One row per (episode, time) step built from the per-step errors of every segment.

    Overlapping segments predict the same step more than once, the entry with the most context (largest position) wins.
    """
    step_error = scores[STEP_ERROR_VAR].values
    step_var_error = scores[STEP_VAR_ERROR_VAR].values
    n_segments, n_steps = step_error.shape
    table = pd.DataFrame(
        {
            episode_coord: np.repeat(scores[episode_coord].values, n_steps),
            "segment": np.repeat(scores["input_batch"].values, n_steps),
            "position": np.tile(np.arange(n_steps), n_segments),
            time_coord: scores[time_coord].values.ravel(),
            "score": step_error.ravel(),
        }
    )
    for i, name in enumerate(scores[VARIABLE_DIM].values):
        table[f"{ERROR_COLUMN_PREFIX}{name}"] = step_var_error[:, :, i].ravel()
    table = table.dropna(subset=["score"])
    table = table.sort_values([episode_coord, time_coord, "position"]).drop_duplicates([episode_coord, time_coord], keep="last")
    return table.reset_index(drop=True)


def _add_flags(table: pd.DataFrame, z_threshold: float, val_episodes: np.ndarray, episode_coord: str) -> pd.DataFrame:
    """Add the robust z-score, the flag, the worst variable and the train/val split to a score table."""
    table = table.copy()
    table["z_score"] = robust_zscore(np.log10(table["score"].values + SCORE_FLOOR))
    table["flagged"] = table["z_score"] > z_threshold
    error_columns = [c for c in table.columns if c.startswith(ERROR_COLUMN_PREFIX)]
    errors = table[error_columns].values
    worst_idx = np.argmax(errors, axis=1)
    table["worst_variable"] = [error_columns[i][len(ERROR_COLUMN_PREFIX) :] for i in worst_idx]
    table["worst_variable_error"] = errors[np.arange(len(table)), worst_idx]
    table["split"] = np.where(np.isin(table[episode_coord].values, val_episodes), "val", "train")
    return table


def _valid_time_per_episode(ds: xr.Dataset, episode_coord: str, time_coord: str) -> pd.Series:
    """Number of non-NaN time steps per episode in the raw dataset."""
    episode_dim, time_dim = episode_and_time_dims(ds, episode_coord, time_coord)
    n_valid = ds[time_coord].notnull().sum(time_dim)
    if episode_dim not in n_valid.dims:
        n_valid = n_valid.broadcast_like(ds[episode_coord])
    return pd.Series(n_valid.values, index=ds[episode_coord].values)


def _episode_summary(table: pd.DataFrame, episode_coord: str, episodes: np.ndarray, n_valid_time: pd.Series | None) -> pd.DataFrame:
    """Per-episode counts of scored and flagged samples, worst z, and (time-independent mode) samples dropped for NaNs."""
    grouped = table.groupby(episode_coord)
    summary = pd.DataFrame(
        {
            "n_scored": grouped.size(),
            "n_flagged": grouped["flagged"].sum(),
            "max_z": grouped["z_score"].max(),
            "mean_score": grouped["score"].mean(),
            "split": grouped["split"].first(),
        }
    )
    summary = summary.reindex(episodes)
    summary["n_scored"] = summary["n_scored"].fillna(0).astype(int)
    summary["n_flagged"] = summary["n_flagged"].fillna(0).astype(int)
    if n_valid_time is not None:
        summary["n_valid_time"] = n_valid_time.reindex(episodes).values
        summary["n_dropped"] = summary["n_valid_time"] - summary["n_scored"]
    summary.index.name = episode_coord
    return summary.sort_values(["n_flagged", "max_z"], ascending=False)


def _write_outputs(table: pd.DataFrame, summary: pd.DataFrame, out_dir: str):
    scores_ds = table.reset_index(drop=True).rename_axis("row").to_xarray()
    scores_ds.to_netcdf(os.path.join(out_dir, SCORES_FILE))
    flagged = table[table["flagged"]].sort_values("z_score", ascending=False)
    flagged.to_csv(os.path.join(out_dir, FLAGGED_FILE), index=False)
    summary.to_csv(os.path.join(out_dir, EPISODE_SUMMARY_FILE))


def _format_table(table: pd.DataFrame, episode_coord: str, show_index: bool) -> str:
    """Tabulate with compact floats, except episode ids which keep every digit."""
    columns = ([table.index.name] if show_index else []) + list(table.columns)
    floatfmt = [".10g" if c == episode_coord else ".4g" for c in columns]
    return tabulate.tabulate(table, headers="keys", showindex=show_index, floatfmt=floatfmt)


def _log_report(table: pd.DataFrame, summary: pd.DataFrame, episode_coord: str, z_threshold: float, out_dir: str):
    n_flagged = int(table["flagged"].sum())
    loguru.logger.info(f"Flagged {n_flagged} of {len(table)} scored samples with z_score > {z_threshold}. Outputs written to {out_dir}.")
    report_columns = [c for c in table.columns if not c.startswith(ERROR_COLUMN_PREFIX)]
    top_flagged = table[table["flagged"]].sort_values("z_score", ascending=False).head(MAX_REPORT_ROWS)[report_columns]
    if len(top_flagged):
        loguru.logger.info(f"Top flagged samples:\n{_format_table(top_flagged, episode_coord, show_index=False)}")
    loguru.logger.info(f"Episodes by number of flags:\n{_format_table(summary.head(MAX_REPORT_ROWS), episode_coord, show_index=True)}")
    unscored = summary.index[summary["n_scored"] == 0].tolist()
    if unscored:
        loguru.logger.warning(f"{len(unscored)} episodes have no scoreable sample (all NaN or too short): {unscored[:MAX_REPORT_ROWS]}")


@dataclass
class AutocheckResult:
    scores: pd.DataFrame  # One row per scored sample with score, z_score, flagged, worst_variable and per-variable errors.
    episode_summary: pd.DataFrame  # One row per episode, sorted by the number of flags.
    trainer: Trainer  # Holds the trained AutocheckScorer in trainer.train_state.model.
    score_dl: DataLoader  # The unshuffled loader every sample was scored with.
    out_dir: str

    def __str__(self) -> str:
        n_flagged = int(self.scores["flagged"].sum())
        return f"Flagged {n_flagged} of {len(self.scores)} samples. Outputs in {self.out_dir}."


def _parse_variables(variables: str | Sequence[str] | None) -> list[str] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        return [name.strip() for name in variables.split(",") if name.strip()]
    return list(variables)


def _build_train_config(mode: str, checkpoint_dir: str, dataloader_config: dict, model_config: dict, train_config: dict) -> dict:
    return {
        "project": "autocheck",
        "train_run_builder": TRAIN_RUN_BUILDER,
        "max_epochs": train_config["max_epochs"],
        "epochs_per_val": 1,
        "checkpoint_dir": checkpoint_dir,
        "patience": train_config["patience"],
        "dataloader_config": dataloader_config,
        "model_init_config": {"mode": mode} | model_config,
        "loss_config": {},
        "optimizer_config": {"learning_rate": train_config["learning_rate"]},
        "test_eval_suite_config": {"time_coord": dataloader_config["time_coord"]},
    }


def autocheck(
    dataset_path: str,
    out_dir: str,
    mode: str = MODE_TIME_INDEP,
    variables: str | Sequence[str] | None = None,
    episode_coord: str = "shot",
    time_coord: str = "time",
    segment_length: int = 40,
    segment_overlap: int = 20,
    val_frac: float = 0.2,
    batch_size: int = 512,
    latent_size: int | None = None,
    explained_variance: float = 0.99,
    width_size: int | None = None,
    depth: int = 2,
    n_heads: int = 2,
    n_blocks: int = 1,
    scaling_type: str = "quantile_50",
    max_epochs: int = 200,
    patience: int = 10,
    learning_rate: float = 1e-3,
    z_threshold: float = 3.5,
    seed: int = 0,
    use_wandb: bool = False,
) -> AutocheckResult:
    """Train a self-supervised model on a tensorized dataset and flag the samples it fits worst.

    Writes autocheck_scores.nc, flagged.csv and episode_summary.csv to out_dir and logs a summary.
    The dataset is loaded into memory. In time-dependent mode memory grows by about segment_length / (segment_length - segment_overlap).

    Args:
        dataset_path (str): Path to a tensorized zarr store, or a netCDF file with the same layout.
        out_dir (str): Directory for the outputs and the model checkpoints.
        mode (str, optional): "time_indep" (autoencoder per sample) or "time_dep" (causal transformer per segment). Defaults to "time_indep".
        variables (str | Sequence[str] | None, optional): Variables to model, comma separated on the command line.
            Defaults to every numeric variable with both the episode and time dimensions.
        episode_coord (str, optional): Name of the episode coordinate. Defaults to "shot".
        time_coord (str, optional): Name of the time coordinate. Defaults to "time".
        segment_length (int, optional): Time steps per segment in time-dependent mode. Defaults to 40.
        segment_overlap (int, optional): Overlapping time steps between consecutive segments. Defaults to 20.
        val_frac (float, optional): Fraction of episodes held out for early stopping. Defaults to 0.2.
        batch_size (int, optional): Samples (or segments) per batch. Defaults to 512.
        latent_size (int | None, optional): Latent size, chosen from the covariance spectrum when None. Defaults to None.
        explained_variance (float, optional): Variance fraction the chosen latent size must explain. Defaults to 0.99.
        width_size (int | None, optional): Hidden width (autoencoder) or model width (transformer), derived when None. Defaults to None.
        depth (int, optional): Hidden layers of the autoencoder MLPs. Defaults to 2.
        n_heads (int, optional): Attention heads of the transformer. Defaults to 2.
        n_blocks (int, optional): Attention blocks of the transformer. Defaults to 1.
        scaling_type (str, optional): Normalization scaling, see popsim.norm_data.ScalingType. Defaults to "quantile_50".
        max_epochs (int, optional): Maximum training epochs. Defaults to 200.
        patience (int, optional): Validation epochs without improvement before early stopping. Defaults to 10.
        learning_rate (float, optional): Adam learning rate. Defaults to 1e-3.
        z_threshold (float, optional): Robust z-score of the log score above which a sample is flagged. Defaults to 3.5.
        seed (int, optional): Seed for the episode split and the model initialization. Defaults to 0.
        use_wandb (bool, optional): Log the training run to Weights & Biases. Defaults to False.

    Returns:
        AutocheckResult: The score table, the per-episode summary, the trainer and the scoring loader.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}.")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, CHECKPOINT_SUBDIR)
    if os.path.exists(checkpoint_dir):
        loguru.logger.warning(f"Removing checkpoints of a previous run in {checkpoint_dir}.")
        shutil.rmtree(checkpoint_dir)

    dataloader_config = {
        "dataset_path": str(dataset_path),
        "mode": mode,
        "variables": _parse_variables(variables),
        "episode_coord": episode_coord,
        "time_coord": time_coord,
        "segment_length": segment_length,
        "segment_overlap": segment_overlap,
        "val_frac": val_frac,
        "batch_size": batch_size,
        "seed": seed,
    }
    model_config = {
        "latent_size": latent_size,
        "explained_variance": explained_variance,
        "width_size": width_size,
        "scaling_type": scaling_type,
        "prng_seed": seed,
    }
    model_config |= {"n_heads": n_heads, "n_blocks": n_blocks} if mode == MODE_TIME_DEP else {"depth": depth}
    train_config = {"max_epochs": max_epochs, "patience": patience, "learning_rate": learning_rate}
    config = _build_train_config(mode, checkpoint_dir, dataloader_config, model_config, train_config)

    trainer, _train_dl, val_dl, score_dl, results = launch_train(config, use_wandb=use_wandb)
    if results is None:
        raise RuntimeError("Training stopped before the scoring pass, no scores were produced.")
    scores = results["test/scores"]

    ds = open_autocheck_dataset(dataset_path, time_coord)
    _, time_dim = episode_and_time_dims(ds, episode_coord, time_coord)
    if mode == MODE_TIME_DEP:
        table = _time_dep_score_table(scores, episode_coord, time_coord)
        n_valid_time = None
    else:
        table = _time_indep_score_table(scores, episode_coord, time_coord, time_dim)
        n_valid_time = _valid_time_per_episode(ds, episode_coord, time_coord)
    val_episodes = np.unique(val_dl.ds[episode_coord].values)
    table = _add_flags(table, z_threshold, val_episodes, episode_coord)
    summary = _episode_summary(table, episode_coord, ds[episode_coord].values, n_valid_time)

    _write_outputs(table, summary, out_dir)
    _log_report(table, summary, episode_coord, z_threshold, out_dir)
    return AutocheckResult(scores=table, episode_summary=summary, trainer=trainer, score_dl=score_dl, out_dir=out_dir)


def main():
    fire.Fire(autocheck)


if __name__ == "__main__":
    main()
