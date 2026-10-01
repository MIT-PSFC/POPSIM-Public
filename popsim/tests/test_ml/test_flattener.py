import numpy as np
import xarray as xr

from popsim.ml.dataloading import make_time_indep_dataloader
from popsim.ml.flattener import VariableFlattener

N_EPISODES, N_TIME = 20, 50
POWER_ON_W = 5e5
GRADIENT_SCALE = 1e21


def _iqr(values: np.ndarray) -> float:
    return np.quantile(values, 0.75) - np.quantile(values, 0.25)


def test_degenerate_features_scaled_by_variable_magnitude():
    """Features without real spread are scaled by their variable's magnitude, the rest keep their plain IQR.

    The power is off in 90 percent of samples, so its IQR is 0 and its scale must be the on value,
    not 1 W, which would put every heated sample at ~5e5 normalized units.
    The gradient's axis feature is pinned to 0 up to 1e-6 noise,
    so its scale must be the gradient's magnitude, not the noise.
    The field spreads 1e-4 of its offset, which must not count as pinned.
    """
    rng = np.random.default_rng(0)
    mask_power_on = rng.random((N_EPISODES, N_TIME)) < 0.1
    power = np.where(mask_power_on, POWER_ON_W, 0.0)
    density = 3e19 * (1.0 + 0.2 * rng.standard_normal((N_EPISODES, N_TIME)))
    field = 5.4 * (1.0 + 1e-4 * rng.standard_normal((N_EPISODES, N_TIME)))
    gradient_axis = 1e-6 * GRADIENT_SCALE * rng.standard_normal((N_EPISODES, N_TIME))
    gradient_mid = GRADIENT_SCALE * rng.standard_normal((N_EPISODES, N_TIME))
    gradient = np.stack([gradient_axis, gradient_mid], axis=-1)
    time = np.broadcast_to(np.arange(N_TIME) * 1e-3, (N_EPISODES, N_TIME))
    ds = xr.Dataset(
        {
            "power": (("episode", "time_idx"), power),
            "density": (("episode", "time_idx"), density),
            "field": (("episode", "time_idx"), field),
            "gradient": (("episode", "time_idx", "rho"), gradient),
        },
        coords={"episode": np.arange(N_EPISODES), "time": (("episode", "time_idx"), time)},
    )
    variables = ["power", "density", "field", "gradient"]
    dl = make_time_indep_dataloader(ds, "time", "episode", variables, variables, convert_xr_to_jnp=False, shuffle=False)

    flattener = VariableFlattener.from_dataloader(dl, variables)

    np.testing.assert_allclose(flattener.scales["power"], POWER_ON_W)
    np.testing.assert_allclose(flattener.scales["density"], _iqr(density), rtol=1e-10)
    np.testing.assert_allclose(flattener.scales["field"], _iqr(field), rtol=1e-8)
    gradient_magnitude = np.percentile(np.abs(gradient - np.median(gradient)), 99)
    np.testing.assert_allclose(flattener.scales["gradient"], [gradient_magnitude, _iqr(gradient_mid)], rtol=1e-10)
    x_flat = flattener.normalize_flat({"power": POWER_ON_W, "density": 3e19, "field": 5.4, "gradient": np.zeros(2)})
    assert np.abs(x_flat[0]) < 1.0
