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


ABERRATION_CODES = {"none": 0, "negate": 1, "scale_4": 2, "zero": 3, "offset": 4, "shuffled": 5}
ABERRATION_CODE_VAR = "aberration_code"
DEMO_SIGNALS = ("signal_x", "signal_y", "signal_a", "signal_b", "signal_c", "signal_d", "signal_e", "signal_f")
# Signals a single aberration event can hit. signal_x and signal_y are left alone,
# signal_y is constant within an episode so negating or zeroing it is often invisible.
ABERRATION_SIGNALS = ("signal_a", "signal_b", "signal_c", "signal_d", "signal_e", "signal_f")
MAX_ABERRATION_STEPS = 3
ABERRATION_OFFSET = 10.0


def _normal_autocheck_episode(episode: int, n_time: int) -> xr.Dataset:
    """One episode of the autocheck demo signals, all functions of the time base and the episode number."""
    time = 0.1 * np.arange(n_time)
    rho = np.linspace(0.0, 1.0, 20)
    R = np.linspace(0.0, 1.0, 8)
    Z = np.linspace(0.0, 1.0, 8)

    # Copy so in-place edits of signal_x never touch the time coordinate.
    signal_x = time.copy()
    signal_y = np.full(n_time, ((3 * episode) + 1) % 2, dtype=float)
    signal_a = np.sqrt(signal_y) + signal_x
    signal_b = 2 * signal_a
    signal_c = np.log(signal_a + 0.1)
    signal_d = ((3 * signal_a) + 1) % 2
    signal_e = signal_a[:, None] * rho[None, :] ** 2
    signal_f = np.log(R[None, :, None] + signal_a[:, None, None] + 0.1) + Z[None, None, :]

    return xr.Dataset(
        data_vars={
            "signal_x": ("time_idx", signal_x),
            "signal_y": ("time_idx", signal_y),
            "signal_a": ("time_idx", signal_a),
            "signal_b": ("time_idx", signal_b),
            "signal_c": ("time_idx", signal_c),
            "signal_d": ("time_idx", signal_d),
            "signal_e": (("time_idx", "rho"), signal_e),
            "signal_f": (("time_idx", "R", "Z"), signal_f),
            ABERRATION_CODE_VAR: ("time_idx", np.zeros(n_time, dtype=int)),
        },
        coords={"time": ("time_idx", time), "rho": rho, "R": R, "Z": Z, "episode": episode},
    )


def _apply_aberration(values: np.ndarray, code: int) -> np.ndarray:
    if code == ABERRATION_CODES["negate"]:
        return -values
    if code == ABERRATION_CODES["scale_4"]:
        return 4 * values
    if code == ABERRATION_CODES["zero"]:
        return 0 * values
    if code == ABERRATION_CODES["offset"]:
        return values + ABERRATION_OFFSET
    raise ValueError(f"Unknown aberration code {code}")


def _add_aberrations(ds: xr.Dataset, n_aberrations: int, rng: np.random.Generator) -> xr.Dataset:
    """Corrupt one signal at a time for a few consecutive steps, n_aberrations times. Events that change nothing are redrawn."""
    n_time = ds.sizes["time_idx"]
    event_codes = [ABERRATION_CODES[name] for name in ("negate", "scale_4", "zero", "offset")]
    n_applied = 0
    while n_applied < n_aberrations:
        signal = ABERRATION_SIGNALS[rng.integers(len(ABERRATION_SIGNALS))]
        code = event_codes[rng.integers(len(event_codes))]
        n_steps = int(rng.integers(1, MAX_ABERRATION_STEPS + 1))
        start = int(rng.integers(0, n_time - n_steps + 1))
        steps = slice(start, start + n_steps)
        before = ds[signal].values[steps]
        after = _apply_aberration(before, code)
        if np.allclose(before, after):
            continue
        ds[signal].values[steps] = after
        ds[ABERRATION_CODE_VAR].values[steps] = code
        n_applied += 1
    return ds


def _shuffle_in_time(ds: xr.Dataset, rng: np.random.Generator) -> xr.Dataset:
    """Permute the rows of every signal in time while the time coordinate stays monotonic."""
    permutation = rng.permutation(ds.sizes["time_idx"])
    for signal in DEMO_SIGNALS:
        ds[signal].values[...] = ds[signal].values[permutation]
    ds[ABERRATION_CODE_VAR].values[...] = ABERRATION_CODES["shuffled"]
    return ds


def generate_autocheck_demo_dataset(
    n_normal: int = 10, n_aberrant: int = 10, n_shuffled: int = 1, n_time: int = 100, n_aberrations: int = 5, seed: int = 0
) -> Generator[xr.Dataset]:
    """Episodes for the dataset autocheck demo, in the per-episode layout build_tensorized_dataset consumes.

    Episodes come in three groups, numbered consecutively:
        1. n_normal episodes of clean signals.
        2. n_aberrant episodes with n_aberrations events each, where one signal is negated, scaled by 4,
           zeroed or offset for a few consecutive steps (time-independent outliers).
        3. n_shuffled episodes of clean signals whose rows are permuted in time (time-dependent outliers).
    The aberration_code variable holds the ground truth per step, see ABERRATION_CODES.
    """
    n_total = n_normal + n_aberrant + n_shuffled
    for episode in range(n_total):
        ds = _normal_autocheck_episode(episode, n_time)
        rng = np.random.default_rng([seed, episode])
        if n_normal <= episode < n_normal + n_aberrant:
            ds = _add_aberrations(ds, n_aberrations, rng)
        elif episode >= n_normal + n_aberrant:
            ds = _shuffle_in_time(ds, rng)
        yield ds
