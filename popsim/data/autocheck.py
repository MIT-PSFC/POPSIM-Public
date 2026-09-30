"""Flag anomalous samples in a tensorized dataset with self-supervised models.

Curating a large dataset by hand is impractical,
and broken or noisy data corrupts training and the reported validation statistics.
This script trains models on the dataset itself and flags the samples they fit worst,
so a human expert only has to review the flagged episodes and times.

Two modes:
    time_indep (default): an MLP autoencoder over every (episode, time) sample, the error is the reconstruction error.
        Catches samples that are off the manifold of the rest of the data.
    time_dep: a causal transformer over segments of consecutive time steps.
        It predicts every variable at step t + 1 from all variables up to step t,
        and the error is the one-step-ahead prediction error per time step.
        Catches transient events and sequences that are individually plausible but inconsistent in time.

Scoring is cross-fitted:
The episodes are split into n_folds folds, and the model of each fold trains on the other folds.
Each model only scores its held-out fold, so no sample is scored by a model that has seen it.

The score of a sample is the mean over variables of each variable's mean squared normalized error,
so a scalar counts as much as a profile or an image.
There is no clean data to train or validate on, so the models train on the robust Cauchy loss of the scores.
Model sizes are derived from the covariance spectrum of the normalized data, see choose_n_latent.
Within each fold the log scores become robust z-scores (median and MAD),
and samples above z_threshold are flagged.

Example:
    popsim-autocheck path/to/dataset.zarr out_dir --mode time_dep --segment_length 40 --segment_overlap 20
"""

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import equinox as eqx
import fire
import jax.numpy as jnp
import loguru
import numpy as np
import optax
import pandas as pd
import scipy.stats
import tabulate
import xarray as xr
from jaxtyping import Array

from popsim.ml import DataLoader, Trainer
from popsim.ml.dataloading import episode_and_time_dims, make_dataloaders
from popsim.ml.envs import ModuleEvalEnvInput
from popsim.ml.eval import eval_model_on_data
from popsim.ml.flattener import VARIABLE_DIM
from popsim.ml.loggers import LoggerBase, NullLogger, WandbLogger
from popsim.ml.split_utils import fracs_to_lengths, split_dataset_by_fracs
from popsim.modules.causal_transformer import CausalTransformerPredictor
from popsim.modules.mlp_autoencoder import MLPAutoEncoder


class AutocheckMode(StrEnum):
    TIME_INDEP = "time_indep"  # MLP autoencoder per (episode, time) sample
    TIME_DEP = "time_dep"  # Causal transformer per segment of consecutive steps


SCORE_FLOOR = 1e-12  # Keeps the log score finite for a perfect fit
MAX_REPORT_ROWS = 30


class AutocheckScorer(eqx.Module):
    """Mean squared normalized error of every variable, the output autocheck trains on and scores with.

    Per sample, var_error is (variable,) for the autoencoder,
    and (time, variable) for the predictor with NaN at steps without a valid prediction.
    """

    model: MLPAutoEncoder | CausalTransformerPredictor

    def __call__(self, inputs: xr.Dataset | ModuleEvalEnvInput) -> dict[str, xr.Variable]:
        flattener = self.model.flattener
        if isinstance(self.model, CausalTransformerPredictor):
            sq_err, mask_valid = self.model.squared_normalized_error(inputs.inputs, inputs.time)
            var_error = flattener.mean_per_variable(sq_err)
            var_error = jnp.where(mask_valid[:, None], var_error, jnp.nan)
            return {"var_error": xr.Variable((self.model.time_dim, VARIABLE_DIM), var_error)}
        sq_err = self.model.squared_normalized_error(inputs)
        var_error = flattener.mean_per_variable(sq_err)
        return {"var_error": xr.Variable((VARIABLE_DIM,), var_error)}

    def trainable(self) -> tuple:
        """The parts to train, normalization statistics stay frozen."""
        return self.model.trainable()


def _var_error_loss(pred: dict[str, xr.Variable], targ: xr.Dataset) -> Array:
    """Robust mean of the per-variable errors over the valid steps, the targets equal the inputs and are unused.

    Every error e enters as the Cauchy loss log(1 + e), which is ~e below one normalized unit and grows only logarithmically above.
    The anomalies in the training and validation episodes then barely steer the fit or the early stopping.
    A segment without a valid step gives 0 instead of NaN.
    """
    var_error = pred["var_error"].data
    mask_valid = ~jnp.isnan(var_error)
    var_error_valid = jnp.where(mask_valid, var_error, 0.0)
    cauchy_loss = jnp.log1p(var_error_valid)
    n_valid = jnp.maximum(mask_valid.sum(), 1)
    return cauchy_loss.sum() / n_valid


def _load_dataset(
    dataset_path: str, variables: Sequence[str] | None, episode_coord: str, time_coord: str, mode: AutocheckMode
) -> xr.Dataset:
    """Open a tensorized zarr store or a netCDF file with the same layout, and load the modeled variables as floats.

    Unless given, the modeled variables are every numeric variable with both the episode and time dims.
    Static variables without a time dim are only allowed in time-independent mode.
    """
    if str(dataset_path).endswith(".zarr"):
        ds = xr.open_zarr(dataset_path)
    else:
        ds = xr.open_dataset(dataset_path)
    # The tensorized layout stores time as a data variable.
    if time_coord in ds.data_vars:
        ds = ds.set_coords(time_coord)
    episode_dim, time_dim = episode_and_time_dims(ds, episode_coord, time_coord)

    if variables is None:
        variables = [
            name
            for name, da in ds.data_vars.items()
            if episode_dim in da.dims and time_dim in da.dims and np.issubdtype(da.dtype, np.number)
        ]
        if not variables:
            raise ValueError("No numeric variables with both the episode and time dimensions found.")
    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Variables {missing} not found in the dataset.")
    static = [name for name in variables if time_dim not in ds[name].dims]
    if static and mode == AutocheckMode.TIME_DEP:
        raise ValueError(f"Variables {static} have no {time_dim} dimension, which time-dependent mode requires.")

    ds = ds[list(variables)]
    for name in variables:
        if not np.issubdtype(ds[name].dtype, np.floating):
            ds[name] = ds[name].astype(float)
    ds = ds.load()
    loguru.logger.info(f"Loaded {len(variables)} variables, {ds.nbytes / 1e9:.2f} GB in memory.")
    return ds


def _fold_loaders(
    ds: xr.Dataset, mask_held_out: np.ndarray, episode_dim: str, val_frac: float, seed: int, loader_kwargs: dict
) -> list[DataLoader]:
    """Train and validation loaders over the episodes outside the held-out fold, and an unshuffled loader over the fold."""
    ds_held_out = ds.isel({episode_dim: mask_held_out})
    ds_rest = ds.isel({episode_dim: ~mask_held_out})
    split_fracs = (1.0 - val_frac, val_frac)
    n_rest = ds_rest.sizes[episode_dim]
    split_lengths = fracs_to_lengths(n_rest, split_fracs)
    if min(split_lengths) == 0:
        raise ValueError(f"val_frac={val_frac} leaves an empty split of the {n_rest} training episodes of a fold, raise it.")
    ds_train, ds_val = split_dataset_by_fracs(ds_rest, split_fracs, episode_dim, seed)
    return make_dataloaders([ds_train, ds_val, ds_held_out], shuffle=[True, False, False], **loader_kwargs)


def _train_scorer(
    train_dl: DataLoader,
    val_dl: DataLoader,
    model_cls: type[MLPAutoEncoder | CausalTransformerPredictor],
    model_kwargs: dict,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    checkpoint_dir: str,
    logger: LoggerBase,
) -> AutocheckScorer:
    """Train a scorer with early stopping on the validation loader and return its best checkpoint."""
    model = model_cls.init(train_dl, **model_kwargs)
    optimizer = optax.adam(learning_rate=learning_rate)
    trainer = Trainer(
        model=AutocheckScorer(model),
        loss_fn=_var_error_loss,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        trainable_getter=AutocheckScorer.trainable,
    )
    trainer.train(train_dl, val_dl, max_epochs=max_epochs, patience=patience, logger=logger)
    trainer.restore_best_checkpoint()
    return trainer.train_state.model


def _held_out_rows(scorer: AutocheckScorer, held_out_dl: DataLoader, episode_coord: str, time_coord: str, fold: int) -> xr.Dataset:
    """The per-variable errors of every scored (episode, time) of the held-out loader, one row each.

    The segments of a time-dependent loader overlap, so a step can be scored more than once.
    The score with the most context, the largest position in its segment, is kept.
    """
    output_ds = eval_model_on_data(scorer, held_out_dl).output_ds
    sample_dim = held_out_dl.metadata.sample_dim
    episode_per_sample = held_out_dl.ds[episode_coord].values
    if held_out_dl.metadata.is_time_dependent:
        time_dim = held_out_dl.metadata.time_dep_metadata.time_dim
        var_error_steps = output_ds["var_error"].transpose(sample_dim, time_dim, VARIABLE_DIM).values
        time_steps = held_out_dl.ds[time_coord].transpose(sample_dim, time_dim).values
        n_samples, n_steps, n_variables = var_error_steps.shape
        var_error_all = var_error_steps.reshape(n_samples * n_steps, n_variables)
        steps = pd.DataFrame(
            {
                "episode": np.repeat(episode_per_sample, n_steps),
                "time": time_steps.reshape(-1),
                "position": np.tile(np.arange(n_steps), n_samples),
            }
        )
        mask_valid = ~np.isnan(var_error_all[:, 0])
        steps_valid = steps[mask_valid].sort_values(["episode", "time", "position"])
        steps_kept = steps_valid.drop_duplicates(["episode", "time"], keep="last")
        var_error_rows = var_error_all[steps_kept.index.values]
        episode_rows = steps_kept["episode"].values
        time_rows = steps_kept["time"].values
    else:
        var_error_rows = output_ds["var_error"].transpose(sample_dim, VARIABLE_DIM).values
        episode_rows = episode_per_sample
        time_rows = held_out_dl.ds[time_coord].values

    fold_rows = np.full(len(episode_rows), fold)
    variables = list(scorer.model.flattener.variables)
    return xr.Dataset(
        {"var_error": (("row", VARIABLE_DIM), var_error_rows)},
        coords={episode_coord: ("row", episode_rows), time_coord: ("row", time_rows), "fold": ("row", fold_rows), VARIABLE_DIM: variables},
    )


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    """z-score with the median and the normal-scaled median absolute deviation, so outliers do not inflate the scale."""
    median = np.nanmedian(x)
    mad = scipy.stats.median_abs_deviation(x, scale="normal", nan_policy="omit")
    mad = max(mad, np.finfo(float).eps)
    return (x - median) / mad


def _with_scores(rows: xr.Dataset) -> xr.Dataset:
    """Add the score of every row and its robust z-score among the rows, which all come from one fold's model."""
    score = rows["var_error"].mean(VARIABLE_DIM)
    log_score = np.log10(score.values + SCORE_FLOOR)
    z_score = _robust_zscore(log_score)
    return rows.assign(score=score, z_score=("row", z_score))


def _with_flags(scores: xr.Dataset, z_threshold: float) -> xr.Dataset:
    """Add the flag and the variable with the largest error to every row."""
    var_error = scores["var_error"]
    worst_variable = var_error.idxmax(VARIABLE_DIM)
    worst_variable_error = var_error.max(VARIABLE_DIM)
    mask_flagged = scores["z_score"] > z_threshold
    return scores.assign(flagged=mask_flagged, worst_variable=worst_variable, worst_variable_error=worst_variable_error)


def _episode_summary(
    rows_df: pd.DataFrame, ds: xr.Dataset, episode_coord: str, time_coord: str, fold_of_episode: np.ndarray
) -> pd.DataFrame:
    """Per-episode counts of scored, unscored and flagged samples, the worst z-score and the fold that scored the episode.

    n_unscored counts the non-NaN times without a score:
    samples with a NaN in a modeled variable, and in time-dependent mode each episode's first step
    and the steps of segments dropped for NaNs.
    """
    episode_dim, time_dim = episode_and_time_dims(ds, episode_coord, time_coord)
    n_time = ds[time_coord].notnull().sum(time_dim)
    if episode_dim not in n_time.dims:
        n_time = n_time.broadcast_like(ds[episode_coord])

    grouped = rows_df.groupby(episode_coord)
    summary = pd.DataFrame(
        {
            "n_scored": grouped.size(),
            "n_flagged": grouped["flagged"].sum(),
            "max_z": grouped["z_score"].max(),
            "mean_score": grouped["score"].mean(),
        }
    )
    summary = summary.reindex(ds[episode_coord].values)
    summary["n_scored"] = summary["n_scored"].fillna(0).astype(int)
    summary["n_flagged"] = summary["n_flagged"].fillna(0).astype(int)
    summary["n_unscored"] = n_time.values - summary["n_scored"]
    summary["fold"] = fold_of_episode
    summary.index.name = episode_coord
    return summary.sort_values(["n_flagged", "max_z"], ascending=False)


def _write_outputs(scores: xr.Dataset, flagged_df: pd.DataFrame, summary: pd.DataFrame, out_dir: str):
    scores.to_netcdf(os.path.join(out_dir, "autocheck_scores.nc"))
    flagged_df.to_csv(os.path.join(out_dir, "flagged.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "episode_summary.csv"))


def _format_table(table: pd.DataFrame, episode_coord: str, show_index: bool) -> str:
    """Tabulate with compact floats, except episode ids which keep every digit."""
    columns = ([table.index.name] if show_index else []) + list(table.columns)
    floatfmt = [".10g" if c == episode_coord else ".4g" for c in columns]
    return tabulate.tabulate(table, headers="keys", showindex=show_index, floatfmt=floatfmt)


def _log_report(
    rows_df: pd.DataFrame, flagged_df: pd.DataFrame, summary: pd.DataFrame, episode_coord: str, z_threshold: float, out_dir: str
):
    loguru.logger.info(
        f"Flagged {len(flagged_df)} of {len(rows_df)} scored samples with z_score > {z_threshold}. Outputs written to {out_dir}."
    )
    if len(flagged_df):
        flagged_top = flagged_df.head(MAX_REPORT_ROWS)
        loguru.logger.info(f"Top flagged samples:\n{_format_table(flagged_top, episode_coord, show_index=False)}")
    summary_top = summary.head(MAX_REPORT_ROWS)
    loguru.logger.info(f"Episodes by number of flags:\n{_format_table(summary_top, episode_coord, show_index=True)}")
    unscored = summary.index[summary["n_scored"] == 0].tolist()
    if unscored:
        loguru.logger.warning(f"{len(unscored)} episodes have no scoreable sample (all NaN or too short): {unscored[:MAX_REPORT_ROWS]}")


@dataclass
class AutocheckResult:
    scores: xr.Dataset  # One row per scored sample with var_error per variable, score, z_score, flagged and the worst variable.
    episode_summary: pd.DataFrame  # One row per episode, sorted by the number of flags.
    scorers: list[AutocheckScorer]  # The trained scorer of every fold, episode_summary holds the fold of each episode.
    out_dir: str

    def __str__(self) -> str:
        n_flagged = int(self.scores["flagged"].sum())
        return f"Flagged {n_flagged} of {self.scores.sizes['row']} samples. Outputs in {self.out_dir}."


def _parse_variables(variables: str | Sequence[str] | None) -> list[str] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        return [name.strip() for name in variables.split(",") if name.strip()]
    return list(variables)


def autocheck(
    dataset_path: str,
    out_dir: str,
    mode: str = AutocheckMode.TIME_INDEP,
    variables: str | Sequence[str] | None = None,
    episode_coord: str = "shot",
    time_coord: str = "time",
    n_folds: int = 5,
    val_frac: float = 0.2,
    segment_length: int = 40,
    segment_overlap: int = 20,
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
    """Train self-supervised models on a tensorized dataset and flag the samples they fit worst.

    One model is trained per fold of episodes and scores only that fold.
    Writes autocheck_scores.nc, flagged.csv and episode_summary.csv to out_dir and logs a summary.
    The dataset is loaded into memory.
    In time-dependent mode memory grows by about segment_length / (segment_length - segment_overlap).

    Args:
        dataset_path (str): Path to a tensorized zarr store, or a netCDF file with the same layout.
        out_dir (str): Directory for the outputs and the model checkpoints.
        mode (str, optional): "time_indep" (autoencoder per sample) or "time_dep" (causal transformer per segment).
            Defaults to "time_indep".
        variables (str | Sequence[str] | None, optional): Variables to model, comma separated on the command line.
            Defaults to every numeric variable with both the episode and time dimensions.
        episode_coord (str, optional): Name of the episode coordinate. Defaults to "shot".
        time_coord (str, optional): Name of the time coordinate. Defaults to "time".
        n_folds (int, optional): Folds of episodes, each scored by a model trained on the others. Defaults to 5.
        val_frac (float, optional): Fraction of a fold's training episodes held out for early stopping. Defaults to 0.2.
        segment_length (int, optional): Time steps per segment in time-dependent mode. Defaults to 40.
        segment_overlap (int, optional): Overlapping time steps between consecutive segments.
            At least 1, so the first step of a segment is scored by the one before it. Defaults to 20.
        batch_size (int, optional): Samples (or segments) per batch. Defaults to 512.
        latent_size (int | None, optional): Latent size, chosen from the covariance spectrum when None. Defaults to None.
        explained_variance (float, optional): Variance fraction the chosen latent size must explain. Defaults to 0.99.
        width_size (int | None, optional): Hidden width (autoencoder) or model width (transformer), derived when None.
            Defaults to None.
        depth (int, optional): Hidden layers of the autoencoder MLPs. Defaults to 2.
        n_heads (int, optional): Attention heads of the transformer. Defaults to 2.
        n_blocks (int, optional): Attention blocks of the transformer. Defaults to 1.
        scaling_type (str, optional): Normalization scaling, see popsim.norm_data.ScalingType. Defaults to "quantile_50".
        max_epochs (int, optional): Maximum training epochs per fold. Defaults to 200.
        patience (int, optional): Validation epochs without improvement before early stopping. Defaults to 10.
        learning_rate (float, optional): Adam learning rate. Defaults to 1e-3.
        z_threshold (float, optional): Robust z-score of the log score above which a sample is flagged. Defaults to 3.5.
        seed (int, optional): Seed for the folds, the validation splits and the model initialization. Defaults to 0.
        use_wandb (bool, optional): Log the training of every fold to Weights & Biases. Defaults to False.

    Returns:
        AutocheckResult: The scores, the per-episode summary and the scorer of every fold.
    """
    mode = AutocheckMode(mode)
    if mode == AutocheckMode.TIME_DEP and not 1 <= segment_overlap < segment_length:
        raise ValueError(f"time_dep needs 1 <= segment_overlap < segment_length, got {segment_overlap} and {segment_length}.")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    if os.path.exists(checkpoint_dir):
        loguru.logger.warning(f"Removing checkpoints of a previous run in {checkpoint_dir}.")
        shutil.rmtree(checkpoint_dir)

    variables = _parse_variables(variables)
    ds = _load_dataset(dataset_path, variables, episode_coord, time_coord, mode)
    variables = list(ds.data_vars)
    episode_dim, _ = episode_and_time_dims(ds, episode_coord, time_coord)
    n_episodes = ds.sizes[episode_dim]
    if not 2 <= n_folds <= n_episodes:
        raise ValueError(f"n_folds must be between 2 and the number of episodes ({n_episodes}), got {n_folds}.")
    rng = np.random.default_rng(seed)
    episode_order = rng.permutation(n_episodes)
    fold_of_episode = episode_order % n_folds

    loader_kwargs = {
        "time_coord": time_coord,
        "episode_coord": episode_coord,
        "input_vars": variables,
        "target_vars": variables,
        "convert_xr_to_jnp": False,
        "batch_size": batch_size,
    }
    model_kwargs = {
        "latent_size": latent_size,
        "explained_variance": explained_variance,
        "width_size": width_size,
        "scaling_type": scaling_type,
    }
    if mode == AutocheckMode.TIME_DEP:
        model_cls = CausalTransformerPredictor
        model_kwargs |= {"n_heads": n_heads, "n_blocks": n_blocks}
        loader_kwargs |= {
            "state_init_vars": [],
            "segment_length": segment_length,
            "segment_overlap": segment_overlap,
            "nan_handling": "drop_segment",
        }
    else:
        model_cls = MLPAutoEncoder
        model_kwargs |= {"depth": depth}

    scorers, rows_per_fold = [], []
    for fold in range(n_folds):
        mask_held_out = fold_of_episode == fold
        loguru.logger.info(f"Fold {fold + 1} of {n_folds}: scoring {mask_held_out.sum()} held-out episodes.")
        train_dl, val_dl, held_out_dl = _fold_loaders(ds, mask_held_out, episode_dim, val_frac, seed + fold, loader_kwargs)
        fold_model_kwargs = model_kwargs | {"prng_seed": seed + fold}
        fold_checkpoint_dir = os.path.join(checkpoint_dir, f"fold_{fold}")
        if use_wandb:
            import wandb

            run_config = fold_model_kwargs | {"mode": str(mode), "fold": fold, "n_folds": n_folds, "learning_rate": learning_rate}
            run = wandb.init(project="autocheck", group=os.path.basename(out_dir), name=f"fold_{fold}", config=run_config)
            logger = WandbLogger(run)
        else:
            logger = NullLogger()
        scorer = _train_scorer(
            train_dl, val_dl, model_cls, fold_model_kwargs, learning_rate, max_epochs, patience, fold_checkpoint_dir, logger
        )
        if use_wandb:
            run.finish()
        fold_rows = _held_out_rows(scorer, held_out_dl, episode_coord, time_coord, fold)
        fold_rows = _with_scores(fold_rows)
        scorers.append(scorer)
        rows_per_fold.append(fold_rows)

    scores = xr.concat(rows_per_fold, dim="row")
    scores = _with_flags(scores, z_threshold)
    rows_df = scores.drop_dims(VARIABLE_DIM).to_dataframe()
    flagged_df = rows_df[rows_df["flagged"]].sort_values("z_score", ascending=False)
    summary = _episode_summary(rows_df, ds, episode_coord, time_coord, fold_of_episode)

    _write_outputs(scores, flagged_df, summary, out_dir)
    _log_report(rows_df, flagged_df, summary, episode_coord, z_threshold, out_dir)
    return AutocheckResult(scores=scores, episode_summary=summary, scorers=scorers, out_dir=out_dir)


def main():
    fire.Fire(autocheck)


if __name__ == "__main__":
    main()
