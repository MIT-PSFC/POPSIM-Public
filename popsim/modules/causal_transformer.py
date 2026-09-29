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
from popsim.modules.mlp_autoencoder import log_latent_choice
from popsim.norm_data import ScalingType
from popsim.utils import time_epsilon

MIN_D_MODEL = 8
MAX_D_MODEL = 128
# Forward-filled padding at the end of a segment gets times that grow by one ulp per step.
PADDING_ULP_FACTOR = 2.0


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

    def valid_steps(self, time: Array) -> Array:
        """Steps that have a prediction and are not forward-filled padding, shape (n_steps,)."""
        dt = jnp.diff(time)
        mask_padded = dt <= PADDING_ULP_FACTOR * time_epsilon(time[:-1])
        return jnp.concatenate([jnp.array([False]), ~mask_padded])

    def predict_next_flat(self, x_seq: Array, time: Array) -> Array:
        """One-step-ahead prediction in normalized flattened units, shape (n_steps, n_flat). Row t predicts x_seq[t + 1]."""
        time_relative = ((time - time[0]) / self.time_scale)[:, None]
        h = jax.vmap(self.embed)(jnp.concatenate([x_seq, time_relative], axis=1)) + self.position_embedding
        for block in self.blocks:
            h = block(h)
        return jax.vmap(self.head)(h)

    def squared_normalized_error(self, inputs: xr.Dataset | dict[str, Array], time: Array) -> tuple[Array, Array]:
        """Elementwise squared prediction error per step, shape (n_steps, n_flat), and the valid step mask.

        The error of step t + 1 is attributed to step t + 1, step 0 gets zeros and is marked invalid.
        """
        x_seq = self.flattener.normalize_flat(inputs)
        x_pred = self.predict_next_flat(x_seq, time)
        sq_err_next = (x_pred[:-1] - x_seq[1:]) ** 2
        sq_err = jnp.concatenate([jnp.zeros((1, x_seq.shape[1])), sq_err_next], axis=0)
        return sq_err, self.valid_steps(time)

    def __call__(self, inputs: xr.Dataset | dict[str, Array], time: Array) -> dict[str, xr.Variable]:
        """One-step-ahead prediction of every variable in physical units, NaN at step 0."""
        x_seq = self.flattener.normalize_flat(inputs)
        x_pred = self.predict_next_flat(x_seq, time)
        x_pred_aligned = jnp.concatenate([jnp.full((1, x_seq.shape[1]), jnp.nan), x_pred[:-1]], axis=0)
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

        The model width is the smallest multiple of n_heads at or above twice the number of principal components
        explaining `explained_variance` of the normalized per-step data, unless given.
        The segment length is fixed by the training loader.
        """
        time_meta = train_dl.metadata.time_dep_metadata
        if time_meta is None:
            raise ValueError("CausalTransformerPredictor needs a time-dependent DataLoader.")
        flattener = VariableFlattener.from_dataloader(train_dl, train_dl.metadata.input_vars, scaling_type)
        n_flat = flattener.n_flat
        n_steps = train_dl.ds.sizes[time_meta.time_dim]

        if latent_size is None:
            rows = flattener.normalized_rows(train_dl.ds, n_covariance_samples, seed=prng_seed)
            latent_size, cumulative_ratio = choose_n_latent(rows, explained_variance)
            log_latent_choice(latent_size, cumulative_ratio, n_flat)
        if width_size is None:
            d_model = math.ceil(2 * latent_size / n_heads) * n_heads
            width_size = int(np.clip(d_model, math.ceil(MIN_D_MODEL / n_heads) * n_heads, (MAX_D_MODEL // n_heads) * n_heads))
        loguru.logger.info(
            f"CausalTransformerPredictor: {n_flat} features, width {width_size}, {n_heads} heads, {n_blocks} blocks, segment length {n_steps}."
        )

        time_scale = _median_time_step(train_dl.ds[time_meta.time_coord].values, time_meta.time_dim, train_dl.ds[time_meta.time_coord].dims)

        key_embed, key_position, key_head, key_blocks = jax.random.split(jax.random.PRNGKey(prng_seed), 4)
        embed = eqx.nn.Linear(n_flat + 1, width_size, key=key_embed)
        position_embedding = 0.02 * jax.random.normal(key_position, (n_steps, width_size))
        blocks = tuple(CausalAttentionBlock(width_size, n_heads, key) for key in jax.random.split(key_blocks, n_blocks))
        head = eqx.nn.Linear(width_size, n_flat, key=key_head)
        return cls(
            flattener=flattener,
            embed=embed,
            position_embedding=position_embedding,
            blocks=blocks,
            head=head,
            time_scale=jnp.asarray(time_scale),
            time_dim=time_meta.time_dim,
        )


def _median_time_step(time: np.ndarray, time_dim: str, dims: tuple[str, ...]) -> float:
    """Median time step over all segments, ignoring the one-ulp steps of forward-filled padding."""
    time_axis = dims.index(time_dim)
    dt = np.diff(time, axis=time_axis)
    time_before = np.take(time, np.arange(time.shape[time_axis] - 1), axis=time_axis)
    mask_real = dt > PADDING_ULP_FACTOR * np.asarray(time_epsilon(time_before))
    if not mask_real.any():
        raise ValueError("No real time steps found in the training segments.")
    return float(np.median(dt[mask_real]))
