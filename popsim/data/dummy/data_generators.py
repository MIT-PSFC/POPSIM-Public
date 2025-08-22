from collections.abc import Generator

import numpy as np
import xarray as xr


def generate_simple_scalar_dataset(n_dataset: int) -> Generator[xr.Dataset]:
    """Build a sequence of datasets with a single scalar variable with increasing length over time.
    Example usage and printout:
        >>> for ds in generate_simple_scalar_dataset(3):
        ...     print(ds)
        ...
        <xarray.Dataset> Size: 24B
        Dimensions:  (time_idx: 1, episode: 1)
        Coordinates:
            time     (time_idx) float64 8B 0.0
        * episode  (episode) int64 8B 0
        Dimensions without coordinates: time_idx
        Data variables:
            var      (time_idx) float64 8B 0.0

        <xarray.Dataset> Size: 40B
        Dimensions:  (time_idx: 2, episode: 1)
        Coordinates:
            time     (time_idx) float64 16B 0.0 1.0
        * episode  (episode) int64 8B 1
        Dimensions without coordinates: time_idx
        Data variables:
            var      (time_idx) float64 16B 0.0 1.0

        <xarray.Dataset> Size: 56B
        Dimensions:  (time_idx: 3, episode: 1)
        Coordinates:
            time     (time_idx) float64 24B 0.0 1.0 2.0
        * episode  (episode) int64 8B 2
        Dimensions without coordinates: time_idx
        Data variables:
            var      (time_idx) float64 24B 0.0 1.0 2.0
    """

    for i in range(n_dataset):
        n_time = i + 1
        ds = xr.Dataset(
            {"var": (("time_idx",), np.arange(n_time, dtype=float))},
            coords={"time": ("time_idx", np.arange(n_time, dtype=float)), "episode": ("episode", [i])},
        )
        yield ds


def generate_dataset_with_spatial_var(episode_identifiers) -> Generator[xr.Dataset]:
    """Build a sequence of datasets with a variable that has a spatial dimension in addition to time."""
    for eps in episode_identifiers:
        nt = np.random.randint(1, 50)
        data = np.random.rand(nt, 5)
        time_series = np.random.rand(nt)
        times = np.random.choice(np.arange(100, dtype=float), size=nt, replace=False)
        times.sort()
        ds = xr.Dataset(
            {"data": (("time_idx", "space"), data), "time_series": (("time_idx"), time_series)},
            coords={"time": ("time_idx", times), "space": ("space", np.arange(5)), "episode": eps},
        )
        yield ds


def generate_dataset_with_changing_spatial_var(n_dataset) -> Generator[xr.Dataset]:
    """Build a sequence of datasets with a variable that has a spatial dimension which is different for each episode.
    Example usage and printout:
    for ds in generate_dataset_with_changing_spatial_var(3):
    ...     print(ds)
    <xarray.Dataset> Size: 64B
    Dimensions:  (time_idx: 3, space_idx: 1, episode: 1)
    Coordinates:
        time     (time_idx) float64 24B 0.0 1.0 2.0
    * episode  (episode) int64 8B 0
        space    (space_idx) float64 8B 0.0
    Dimensions without coordinates: time_idx, space_idx
    Data variables:
        var      (time_idx, space_idx) float64 24B 0.0 0.0 0.0
    <xarray.Dataset> Size: 96B
    Dimensions:  (time_idx: 3, space_idx: 2, episode: 1)
    Coordinates:
        time     (time_idx) float64 24B 0.0 1.0 2.0
    * episode  (episode) int64 8B 1
        space    (space_idx) float64 16B 0.0 1.0
    Dimensions without coordinates: time_idx, space_idx
    Data variables:
        var      (time_idx, space_idx) float64 48B 0.0 1.0 0.0 1.0 0.0 1.0
    <xarray.Dataset> Size: 128B
    Dimensions:  (time_idx: 3, space_idx: 3, episode: 1)
    Coordinates:
        time     (time_idx) float64 24B 0.0 1.0 2.0
    * episode  (episode) int64 8B 2
        space    (space_idx) float64 24B 0.0 1.0 2.0
    Dimensions without coordinates: time_idx, space_idx
    Data variables:
        var      (time_idx, space_idx) float64 72B 0.0 1.0 2.0 0.0 ... 0.0 1.0 2.0
    """
    n_time = 3
    times = np.arange(n_time, dtype=float)
    for i in range(n_dataset):
        n_space = i + 1
        data = np.tile(np.arange(n_space, dtype=float), (n_time, 1))
        ds = xr.Dataset(
            data_vars={"var": (("time_idx", "space_idx"), data)},
            coords={"time": ("time_idx", times), "episode": ("episode", [i]), "space": ("space_idx", np.arange(n_space, dtype=float))},
        )
        yield ds
