import jax.numpy as jnp
import numpy as np

from popsim.data.dummy.data_generators import DEMO_SIGNALS
from popsim.ml.dataloading import make_time_dep_dataloader
from popsim.ml.eval import eval_model_on_data
from popsim.ml.utils import pad_time_with_epsilon
from popsim.modules.causal_transformer import CausalTransformerPredictor
from popsim.tests.test_modules.test_mlp_autoencoder import demo_dataset


def _model_and_loader(segment_length: int = 10):
    ds = demo_dataset(n_time=30)
    variables = list(DEMO_SIGNALS)
    dl = make_time_dep_dataloader(
        ds, "time", "episode", [], variables, variables, convert_xr_to_jnp=False,
        segment_length=segment_length, segment_overlap=segment_length // 2, batch_size=None, shuffle=False,
    )
    model = CausalTransformerPredictor.init(dl, latent_size=3, n_heads=2, n_blocks=2)
    return model, dl


def test_predictions_are_causal():
    model, dl = _model_and_loader()
    env_input, _ = next(iter(dl)).get_inputs_and_targets()
    inputs_first = env_input.inputs.isel(sample=0)
    time_first = jnp.asarray(env_input.time[0])

    x_seq = model.flattener.normalize_flat(inputs_first)
    pred = model.predict_next_flat(x_seq, time_first)
    perturbed_step = 4
    pred_perturbed = model.predict_next_flat(x_seq.at[perturbed_step].add(1.0), time_first)

    np.testing.assert_allclose(pred_perturbed[:perturbed_step], pred[:perturbed_step])
    assert not np.allclose(pred_perturbed[perturbed_step], pred[perturbed_step])


def test_valid_steps_masks_forward_filled_padding():
    model, _ = _model_and_loader()
    time = pad_time_with_epsilon(jnp.array([0.0, 0.1, 0.2, 0.3, 0.3, 0.3]))
    np.testing.assert_array_equal(model.valid_steps(time), [False, True, True, True, False, False])


def test_output_dims_and_error_shapes():
    model, dl = _model_and_loader(segment_length=10)
    env_input, _ = next(iter(dl)).get_inputs_and_targets()
    inputs_first = env_input.inputs.isel(sample=0)
    sq_err, mask_valid = model.squared_normalized_error(inputs_first, jnp.asarray(env_input.time[0]))
    assert sq_err.shape == (10, model.flattener.n_flat)
    assert mask_valid.shape == (10,)
    assert not mask_valid[0]
    np.testing.assert_array_equal(sq_err[0], 0.0)

    output_ds = eval_model_on_data(lambda env: model(env.inputs, env.time), dl).output_ds
    assert output_ds["signal_f"].dims == ("sample", "time_idx_input", "R", "Z")
    assert output_ds["signal_a"].dims == ("sample", "time_idx_input")
    # Step 0 has no prediction, later steps are finite.
    assert np.all(np.isnan(output_ds["signal_a"].isel(time_idx_input=0).values))
    assert np.all(np.isfinite(output_ds["signal_a"].isel(time_idx_input=slice(1, None)).values))
