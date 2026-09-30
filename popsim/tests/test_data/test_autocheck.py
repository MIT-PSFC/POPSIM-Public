import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from popsim.data.autocheck import AutocheckResult, autocheck
from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.data.dummy.data_generators import (
    ABERRATION_CODE_VAR,
    DEMO_SIGNALS,
    EVENT_ABERRATIONS,
    Aberration,
    generate_autocheck_demo_dataset,
)

N_NORMAL, N_ABERRANT, N_SHUFFLED, N_TIME = 40, 3, 1, 80
SHUFFLED_EPISODE = N_NORMAL + N_ABERRANT
NAN_HOLE_EPISODE, NAN_HOLE_STEPS = 2, slice(10, 14)
N_FOLDS = 3


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
    truth = ds[[ABERRATION_CODE_VAR, "time"]].to_dataframe().reset_index()
    return zarr_path, nc_path, truth[["episode", "time", ABERRATION_CODE_VAR]]


def _scores_with_truth(result: AutocheckResult, truth: pd.DataFrame) -> pd.DataFrame:
    scores = result.scores.drop_dims("variable").to_dataframe()
    return scores.merge(truth, on=["episode", "time"], how="left")


def test_time_indep_flags_aberrations(demo_paths):
    _, nc_path, truth = demo_paths
    result = autocheck(
        nc_path, os.path.join(os.path.dirname(nc_path), "out_indep"), mode="time_indep", variables=list(DEMO_SIGNALS),
        episode_coord="episode", n_folds=N_FOLDS, max_epochs=200, patience=40, batch_size=64, seed=0,
    )
    scores = _scores_with_truth(result, truth)
    mask_aberrant = scores[ABERRATION_CODE_VAR].isin(EVENT_ABERRATIONS)
    mask_normal = scores[ABERRATION_CODE_VAR] == Aberration.NONE
    # Aberrant steps score well above clean ones, and a good share of them crosses the flag threshold.
    # Some injected events are subtle (e.g. zeroing a signal that is already near zero), so the recall bound is moderate.
    assert scores.loc[mask_aberrant, "z_score"].median() > scores.loc[mask_normal, "z_score"].median() + 1.5
    assert scores.loc[mask_aberrant, "flagged"].mean() > 0.3
    assert scores.loc[mask_normal, "flagged"].mean() < 0.01

    summary = result.episode_summary
    assert summary.loc[NAN_HOLE_EPISODE, "n_unscored"] == NAN_HOLE_STEPS.stop - NAN_HOLE_STEPS.start
    assert (summary.drop(NAN_HOLE_EPISODE)["n_unscored"] == 0).all()


def test_time_dep_flags_shuffled_episode(demo_paths):
    zarr_path, _, truth = demo_paths
    result = autocheck(
        zarr_path, os.path.join(os.path.dirname(zarr_path), "out_dep"), mode="time_dep", variables=list(DEMO_SIGNALS),
        episode_coord="episode", n_folds=N_FOLDS, segment_length=20, segment_overlap=10, max_epochs=200, patience=30,
        batch_size=8, seed=0,
    )
    scores = _scores_with_truth(result, truth)
    mean_z = scores.groupby("episode")["z_score"].mean()
    clean_episodes = list(range(N_NORMAL))
    assert mean_z[SHUFFLED_EPISODE] > mean_z[clean_episodes].max()
    assert scores.loc[scores["episode"] == SHUFFLED_EPISODE, "flagged"].mean() > 0.8
    mask_normal = scores[ABERRATION_CODE_VAR] == Aberration.NONE
    assert scores.loc[mask_normal, "flagged"].mean() < 0.05
    # Overlapping segments score every step but each episode's first exactly once.
    assert not scores.duplicated(["episode", "time"]).any()
    assert (result.episode_summary["n_unscored"] == 1).all()
