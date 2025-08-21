from collections.abc import Iterator

import numpy as np
import xarray as xr


def iterate_simple_scalar_dataset(n_dataset: int) -> Iterator[xr.Dataset]:
    """Build a sequence of datasets with a single scalar variable with increasing length over time.
    Example usage and printout:
        >>> for ds in iterate_simple_scalar_dataset(3):
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
