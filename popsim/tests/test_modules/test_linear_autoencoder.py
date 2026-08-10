import os

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr

from popsim import DATA_DIR
from popsim.modules.linear_autoencoder import LinearAutoEncoder


def test_linear_encoder():
    ds = xr.open_dataset(os.path.join(DATA_DIR, "dummy/torax_profile_predictor.nc")).stack(sample=("simulation", "time")).transpose("sample", ...)
    n_latent = 20

    # Test fitting a single LinearAutoEncoder works, and that the round-trip error is small.
    model, _ = LinearAutoEncoder.fit(ds["temp_el"].data, n_latent)
    def roundtrip_sample(samp):
        encoded = model.forward(samp)
        decoded = model.backward(encoded)
        relative_error = jnp.linalg.norm(decoded - samp) / jnp.linalg.norm(samp)
        return relative_error

    relative_errors = jax.vmap(roundtrip_sample)(ds["temp_el"].data)

    # Expect a low error for the round-trip, but non-zero because of dimensionality reduction.
    assert relative_errors.mean() < 1e-3


    # Test fit_tree works and that the round-trip error is small.
    data_dict = {"ne": ds["ne"].data, "ni": ds["ni"].data}
    linear_encoders = LinearAutoEncoder.fit_tree(data_dict, n_latent)
    assert linear_encoders["ne"].components.shape == (ds.sizes["rho_cell_norm"], n_latent)
    assert linear_encoders["ne"].means.shape == (ds.sizes["rho_cell_norm"],)
    assert linear_encoders["ne"].scaling.shape == (ds.sizes["rho_cell_norm"],)
    assert linear_encoders["ni"].components.shape == (ds.sizes["rho_cell_norm"], n_latent)
    assert linear_encoders["ni"].means.shape == (ds.sizes["rho_cell_norm"],)
    assert linear_encoders["ni"].scaling.shape == (ds.sizes["rho_cell_norm"],)

    def roundtrip_tree_sample(samp_dict):
        encoded_dict = {k: enc.forward(samp_dict[k]) for k, enc in linear_encoders.items()}
        decoded_dict = {k: enc.backward(encoded_dict[k]) for k, enc in linear_encoders.items()}
        relative_error_ne = jnp.linalg.norm(decoded_dict["ne"] - samp_dict["ne"]) / jnp.linalg.norm(samp_dict["ne"])
        relative_error_ni = jnp.linalg.norm(decoded_dict["ni"] - samp_dict["ni"]) / jnp.linalg.norm(samp_dict["ni"])
        return relative_error_ne, relative_error_ni
    relative_errors_ne, relative_errors_ni = jax.vmap(roundtrip_tree_sample)(data_dict)
    assert relative_errors_ne.mean() < 1e-3
    assert relative_errors_ni.mean() < 1e-3
    
    # Test fit_flattened_tree works and that the round-trip error is small.
    linear_encoder2, _ = LinearAutoEncoder.fit_flattened_tree(data_dict, n_latent)
    assert linear_encoder2.means.size == ds.sizes["rho_cell_norm"] * 2
    assert linear_encoder2.scaling.size == ds.sizes["rho_cell_norm"] * 2
    assert linear_encoder2.components.shape == (ds.sizes["rho_cell_norm"] * 2, n_latent)
    data_flat = jax.vmap(lambda x: jax.flatten_util.ravel_pytree(x)[0])(data_dict)
    def roundtrip_flattened_tree_sample(samp_flat):
        encoded = linear_encoder2.forward(samp_flat)
        decoded = linear_encoder2.backward(encoded)
        relative_error = jnp.linalg.norm(decoded - samp_flat) / jnp.linalg.norm(samp_flat)
        return relative_error
    relative_errors_flat = jax.vmap(roundtrip_flattened_tree_sample)(data_flat)
    assert relative_errors_flat.mean() < 1e-3
