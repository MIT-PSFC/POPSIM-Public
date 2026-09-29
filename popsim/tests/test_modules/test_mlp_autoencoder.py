import jax
import numpy as np
import xarray as xr

from popsim.data.dummy.data_generators import DEMO_SIGNALS, generate_autocheck_demo_dataset
from popsim.ml.dataloading import make_time_indep_dataloader
from popsim.ml.eval import eval_model_on_data
from popsim.modules.linear_autoencoder import choose_n_latent
from popsim.modules.mlp_autoencoder import MLPAutoEncoder
from popsim.xarray_utils import run_function_with_dim_removed


def demo_dataset(n_time: int = 30) -> xr.Dataset:
    """A few clean demo episodes in memory, with a 2D time coordinate like the tensorized layout."""
    episodes = list(generate_autocheck_demo_dataset(n_normal=3, n_aberrant=0, n_shuffled=0, n_time=n_time))
    ds = xr.concat(episodes, dim="episode")
    time_2d = np.broadcast_to(ds["time"].values, (ds.sizes["episode"], n_time))
    return ds.assign_coords(time=(("episode", "time_idx"), time_2d))


def test_choose_n_latent_recovers_rank():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 12))
    n_latent, cumulative_ratio = choose_n_latent(rows, explained_variance=0.99)
    assert n_latent == 3
    np.testing.assert_allclose(cumulative_ratio[2:], 1.0, atol=1e-8)


def test_mlp_autoencoder_shapes_and_output_dims():
    ds = demo_dataset()
    variables = list(DEMO_SIGNALS)
    dl = make_time_indep_dataloader(ds, "time", "episode", variables, variables, convert_xr_to_jnp=False, batch_size=32, shuffle=False)
    model = MLPAutoEncoder.init(dl, latent_size=3, width_size=16)

    # 6 scalars, a 20 point profile and an 8 x 8 image per sample.
    assert model.flattener.n_flat == 6 + 20 + 64
    assert model.flattener.shape_of("signal_f") == (8, 8)

    inputs, _ = next(iter(dl)).get_inputs_and_targets()
    sq_err = run_function_with_dim_removed(model.squared_normalized_error, (inputs,), "sample", in_axes=(0,))
    assert sq_err.shape == (32, model.flattener.n_flat)
    assert np.all(np.isfinite(sq_err))

    output_ds = eval_model_on_data(model, dl).output_ds
    assert output_ds["signal_f"].dims == ("sample", "R", "Z")
    assert output_ds["signal_e"].dims == ("sample", "rho")
    assert output_ds["signal_a"].dims == ("sample",)
    assert output_ds.sizes["sample"] == dl.dataset.n_samples

    # Only the encoder and decoder train, the normalization statistics are left out.
    trainable_ids = {id(leaf) for leaf in jax.tree.leaves(model.trainable())}
    assert id(model.flattener.means["signal_a"]) not in trainable_ids
    assert id(model.encoder.layers[0].weight) in trainable_ids
