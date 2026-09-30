import math

import equinox as eqx
import jax
import jax.numpy as jnp
import loguru
import numpy as np
import xarray as xr
from jaxtyping import Array

from popsim.ml.dataloading import DataLoader
from popsim.ml.flattener import VariableFlattener
from popsim.modules.linear_autoencoder import choose_n_latent
from popsim.norm_data import ScalingType
from popsim.utils import time_epsilon

MIN_D_MODEL = 8
MAX_D_MODEL = 128
# Forward-filled padding at the end of a segment gets times that grow by one ulp per step.
PADDING_ULP_FACTOR = 2.0


def mask_valid_steps(time: Array) -> Array:
    """Steps of a segment that have a prediction and are not forward-filled padding, shape (n_steps,).

    Step 0 has nothing to predict from.
    """
    dt = jnp.diff(time)
    mask_padded = dt <= PADDING_ULP_FACTOR * time_epsilon(time[:-1])
    return jnp.concatenate([jnp.array([False]), ~mask_padded])


class CausalAttentionBlock(eqx.Module):
    """Pre-norm transformer block with a causal self-attention mask."""

    attention: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm_attention: eqx.nn.LayerNorm
    norm_mlp: eqx.nn.LayerNorm

    def __init__(self, d_model: int, n_heads: int, key: jax.random.PRNGKey):
        key_attention, key_mlp = jax.random.split(key)
        self.attention = eqx.nn.MultiheadAttention(num_heads=n_heads, query_size=d_model, key=key_attention)
        self.mlp = eqx.nn.MLP(d_model, d_model, width_size=2 * d_model, depth=1, activation=jax.nn.gelu, key=key_mlp)
        self.norm_attention = eqx.nn.LayerNorm(d_model)
        self.norm_mlp = eqx.nn.LayerNorm(d_model)

    def __call__(self, x: Array) -> Array:
        n_steps = x.shape[0]
        mask_causal = jnp.tril(jnp.ones((n_steps, n_steps), dtype=bool))
        h = jax.vmap(self.norm_attention)(x)
        x = x + self.attention(h, h, h, mask=mask_causal)
        h = jax.vmap(self.norm_mlp)(x)
        x = x + jax.vmap(self.mlp)(h)
        return x


class CausalTransformerPredictor(eqx.Module):
    """Predict every variable at the next time step from all variables up to the current step.

    Works on one segment of consecutive time steps with dims (time, *feature) per variable.
    Row t of the prediction is the estimate of step t + 1, so step 0 has no prediction.
    """

    flattener: VariableFlattener
    embed: eqx.nn.Linear
    position_embedding: Array
    blocks: tuple[CausalAttentionBlock, ...]
    head: eqx.nn.Linear
    time_scale: Array
    time_dim: str = eqx.field(static=True)

    def predict_next_flat(self, x_seq: Array, time: Array) -> Array:
        """One-step-ahead prediction in normalized flattened units, shape (n_steps, n_flat).

        Row t predicts x_seq[t + 1].
        """
        time_relative = (time - time[0]) / self.time_scale
        x_with_time = jnp.concatenate([x_seq, time_relative[:, None]], axis=1)
        h = jax.vmap(self.embed)(x_with_time) + self.position_embedding
        for block in self.blocks:
            h = block(h)
        return jax.vmap(self.head)(h)

    def squared_normalized_error(self, inputs: xr.Dataset | dict[str, Array], time: Array) -> tuple[Array, Array]:
        """Elementwise squared prediction error per step, shape (n_steps, n_flat), and the valid step mask.

        The error of the prediction of step t + 1 is attributed to step t + 1.
        Step 0 gets zeros and is marked invalid.
        """
        x_seq = self.flattener.normalize_flat(inputs)
        x_pred = self.predict_next_flat(x_seq, time)
        sq_err_next = (x_pred[:-1] - x_seq[1:]) ** 2
        sq_err_first = jnp.zeros((1, x_seq.shape[1]))
        sq_err = jnp.concatenate([sq_err_first, sq_err_next], axis=0)
        return sq_err, mask_valid_steps(time)

    def __call__(self, inputs: xr.Dataset | dict[str, Array], time: Array) -> dict[str, xr.Variable]:
        """One-step-ahead prediction of every variable in physical units, NaN at step 0."""
        x_seq = self.flattener.normalize_flat(inputs)
        x_pred = self.predict_next_flat(x_seq, time)
        x_pred_first = jnp.full((1, x_seq.shape[1]), jnp.nan)
        x_pred_aligned = jnp.concatenate([x_pred_first, x_pred[:-1]], axis=0)
        pred = self.flattener.unflatten_unnorm(x_pred_aligned)
        return {
            name: xr.Variable(dims=(self.time_dim, *self.flattener.dims_of(name)), data=pred[name]) for name in self.flattener.variables
        }

    def trainable(self) -> tuple:
        """The parts to train, normalization statistics and the time scale stay frozen."""
        return (self.embed, self.position_embedding, self.blocks, self.head)

    @classmethod
    def init(
        cls,
        train_dl: DataLoader,
        latent_size: int | None = None,
        explained_variance: float = 0.99,
        width_size: int | None = None,
        n_heads: int = 2,
        n_blocks: int = 1,
        scaling_type: ScalingType | str = ScalingType.QUANTILE_50,
        n_covariance_samples: int = 10_000,
        prng_seed: int = 0,
    ) -> "CausalTransformerPredictor":
        """Build a predictor sized from the training data.

        Unless given, the model width is the smallest multiple of n_heads at or above twice the latent size,
        where the latent size is the number of principal components
        explaining `explained_variance` of the normalized per-step data.
        The segment length is fixed by the training loader.
        """
        time_meta = train_dl.metadata.time_dep_metadata
        if time_meta is None:
            raise ValueError("CausalTransformerPredictor needs a time-dependent DataLoader.")
        flattener = VariableFlattener.from_dataloader(train_dl, train_dl.metadata.input_vars, scaling_type)
        n_flat = flattener.n_flat
        n_steps = train_dl.ds.sizes[time_meta.time_dim]

        if latent_size is None:
            rows = flattener.normalized_rows(train_dl, n_covariance_samples, seed=prng_seed)
            latent_size = choose_n_latent(rows, explained_variance)
        if width_size is None:
            width_min = math.ceil(MIN_D_MODEL / n_heads) * n_heads
            width_max = (MAX_D_MODEL // n_heads) * n_heads
            width_twice_latent = math.ceil(2 * latent_size / n_heads) * n_heads
            width_size = int(np.clip(width_twice_latent, width_min, width_max))
        loguru.logger.info(
            f"CausalTransformerPredictor: {n_flat} features, width {width_size}, {n_heads} heads, {n_blocks} blocks, segment length {n_steps}."
        )

        time_segments = train_dl.ds[time_meta.time_coord].transpose(train_dl.metadata.sample_dim, time_meta.time_dim)
        time_scale = _median_time_step(jnp.asarray(time_segments.values))

        key = jax.random.PRNGKey(prng_seed)
        key_embed, key_position, key_head, key_blocks = jax.random.split(key, 4)
        embed = eqx.nn.Linear(n_flat + 1, width_size, key=key_embed)
        position_embedding = 0.02 * jax.random.normal(key_position, (n_steps, width_size))
        keys_block = jax.random.split(key_blocks, n_blocks)
        blocks = tuple(CausalAttentionBlock(width_size, n_heads, key_block) for key_block in keys_block)
        head = eqx.nn.Linear(width_size, n_flat, key=key_head)
        return cls(
            flattener=flattener,
            embed=embed,
            position_embedding=position_embedding,
            blocks=blocks,
            head=head,
            time_scale=time_scale,
            time_dim=time_meta.time_dim,
        )


def _median_time_step(time_segments: Array) -> Array:
    """Median time step over segments of shape (n_segments, n_steps), ignoring forward-filled padding."""
    dt = jnp.diff(time_segments, axis=1)
    mask_valid = jax.vmap(mask_valid_steps)(time_segments)
    # Step t + 1 is valid exactly when the step from t to t + 1 is real, so the mask aligns with dt.
    mask_real_dt = mask_valid[:, 1:]
    if not mask_real_dt.any():
        raise ValueError("No real time steps found in the training segments.")
    dt_real = dt[mask_real_dt]
    return jnp.median(dt_real)
