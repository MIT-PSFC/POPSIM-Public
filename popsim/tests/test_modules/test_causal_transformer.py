import jax.numpy as jnp
import numpy as np
import xarray as xr

from popsim.data.dummy.data_generators import DEMO_SIGNALS, generate_autocheck_demo_dataset
from popsim.ml.dataloading import make_time_dep_dataloader
from popsim.ml.utils import pad_time_with_epsilon
from popsim.modules.causal_transformer import CausalTransformerPredictor, mask_valid_steps


def _demo_dataset(n_time: int = 30) -> xr.Dataset:
    """A few clean demo episodes in memory, with a 2D time coordinate like the tensorized layout."""
    episodes = list(generate_autocheck_demo_dataset(n_normal=3, n_aberrant=0, n_shuffled=0, n_time=n_time))
    ds = xr.concat(episodes, dim="episode")
    time_2d = np.broadcast_to(ds["time"].values, (ds.sizes["episode"], n_time))
    return ds.assign_coords(time=(("episode", "time_idx"), time_2d))


def test_predictions_are_causal():
    ds = _demo_dataset(n_time=30)
    variables = list(DEMO_SIGNALS)
    dl = make_time_dep_dataloader(
        ds, "time", "episode", [], variables, variables, convert_xr_to_jnp=False,
        segment_length=10, segment_overlap=5, batch_size=None, shuffle=False,
    )
    model = CausalTransformerPredictor.init(dl, latent_size=3, n_heads=2, n_blocks=2)
    env_input, _ = next(iter(dl)).get_inputs_and_targets()
    inputs_first = env_input.inputs.isel(sample=0)
    time_first = jnp.asarray(env_input.time[0])

    x_seq = model.flattener.normalize_flat(inputs_first)
    pred = model.predict_next_flat(x_seq, time_first)
    perturbed_step = 4
    pred_perturbed = model.predict_next_flat(x_seq.at[perturbed_step].add(1.0), time_first)

    np.testing.assert_allclose(pred_perturbed[:perturbed_step], pred[:perturbed_step])
    assert not np.allclose(pred_perturbed[perturbed_step], pred[perturbed_step])


def test_mask_valid_steps_excludes_forward_filled_padding():
    time = pad_time_with_epsilon(jnp.array([0.0, 0.1, 0.2, 0.3, 0.3, 0.3]))
    np.testing.assert_array_equal(mask_valid_steps(time), [False, True, True, True, False, False])
