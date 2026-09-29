import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from popsim.data.autocheck import EPISODE_SUMMARY_FILE, FLAGGED_FILE, SCORES_FILE, autocheck, robust_zscore
from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.data.dummy.data_generators import ABERRATION_CODE_VAR, ABERRATION_CODES, DEMO_SIGNALS, generate_autocheck_demo_dataset

N_NORMAL, N_ABERRANT, N_SHUFFLED, N_TIME = 6, 3, 1, 80
SHUFFLED_EPISODE = N_NORMAL + N_ABERRANT
NAN_HOLE_EPISODE, NAN_HOLE_STEPS = 2, slice(10, 14)


@pytest.fixture(scope="module")
def demo_paths(tmp_path_factory):
    """The demo dataset as a zarr store, plus a netCDF copy with a NaN hole in one episode."""
    work_dir = tmp_path_factory.mktemp("autocheck")
    zarr_path = str(work_dir / "demo.zarr")
    episodes = generate_autocheck_demo_dataset(n_normal=N_NORMAL, n_aberrant=N_ABERRANT, n_shuffled=N_SHUFFLED, n_time=N_TIME)
    build_tensorized_dataset(lambda ds: ds, episodes, zarr_path, time_dim="time_idx", episode_dim="episode", mb_per_chunk=None)

    ds = xr.open_zarr(zarr_path).load()
    ds["signal_a"][NAN_HOLE_EPISODE, NAN_HOLE_STEPS] = np.nan
    nc_path = str(work_dir / "demo_with_hole.nc")
    ds.to_netcdf(nc_path)
    # time is a data variable in the tensorized layout, keep it so time-dependent scores can be joined on it.
    truth = ds[[ABERRATION_CODE_VAR, "time"]].to_dataframe().reset_index()
    return zarr_path, nc_path, truth


def _with_ground_truth(scores: pd.DataFrame, truth: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    return scores.merge(truth[[*on, ABERRATION_CODE_VAR]], on=on, how="left")


def test_robust_zscore_ignores_outliers():
    x = np.concatenate([np.zeros(50), np.ones(50), [1000.0]])
    z = robust_zscore(x)
    assert abs(z[:100]).max() < 2.0
    assert z[-1] > 100.0


def test_time_indep_flags_aberrations(demo_paths):
    _, nc_path, truth = demo_paths
    result = autocheck(
        nc_path, os.path.join(os.path.dirname(nc_path), "out_indep"), mode="time_indep", variables=list(DEMO_SIGNALS),
        episode_coord="episode", max_epochs=200, patience=40, batch_size=64, seed=0,
    )
    scores = _with_ground_truth(result.scores, truth, on=["episode", "time_idx"])
    mask_aberrant = scores[ABERRATION_CODE_VAR].between(1, 4)
    mask_normal = scores[ABERRATION_CODE_VAR] == ABERRATION_CODES["none"]
    # Aberrant steps score well above clean ones, and a good share of them crosses the flag threshold.
    # Some injected events are subtle (e.g. zeroing a signal that is already near zero), so the margin is moderate.
    assert scores.loc[mask_aberrant, "z_score"].median() > scores.loc[mask_normal, "z_score"].median() + 1.5
    assert scores.loc[mask_aberrant, "flagged"].mean() > 0.25
    assert scores.loc[mask_normal, "flagged"].mean() < 0.05

    summary = result.episode_summary
    assert summary.loc[NAN_HOLE_EPISODE, "n_dropped"] == NAN_HOLE_STEPS.stop - NAN_HOLE_STEPS.start
    assert (summary.drop(NAN_HOLE_EPISODE)["n_dropped"] == 0).all()
    assert set(summary["split"]) == {"train", "val"}
    for name in (SCORES_FILE, FLAGGED_FILE, EPISODE_SUMMARY_FILE):
        assert os.path.exists(os.path.join(result.out_dir, name))
    written = xr.open_dataset(os.path.join(result.out_dir, SCORES_FILE))
    assert written.sizes["row"] == len(result.scores)


def test_time_dep_flags_shuffled_episode(demo_paths):
    zarr_path, _, truth = demo_paths
    result = autocheck(
        zarr_path, os.path.join(os.path.dirname(zarr_path), "out_dep"), mode="time_dep", variables=list(DEMO_SIGNALS),
        episode_coord="episode", segment_length=20, segment_overlap=10, max_epochs=200, patience=30, batch_size=8, seed=0,
    )
    scores = _with_ground_truth(result.scores, truth, on=["episode", "time"])
    mean_z = scores.groupby("episode")["z_score"].mean()
    clean_episodes = range(N_NORMAL)
    assert mean_z[SHUFFLED_EPISODE] > mean_z[list(clean_episodes)].max()
    assert scores.loc[scores["episode"] == SHUFFLED_EPISODE, "flagged"].mean() > 0.3
    mask_normal_step = scores[ABERRATION_CODE_VAR] == ABERRATION_CODES["none"]
    assert scores.loc[mask_normal_step, "flagged"].mean() < 0.1
    # Every scored step is a unique (episode, time) even though segments overlap.
    assert not scores.duplicated(["episode", "time"]).any()
