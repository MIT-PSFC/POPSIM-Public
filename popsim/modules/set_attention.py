import equinox as eqx
import jax
from jaxtyping import Array

from popsim import TimeIndepModule


class SetAttentionBlock(TimeIndepModule):
    """Set attention block as defined in:

    Lee, Juho, et al. "Set transformer: A framework for attention-based permutation-invariant neural networks." International conference on machine learning. PMLR, 2019.
    """

    multiheaded_attention: eqx.nn.MultiheadAttention
    output_layer: eqx.nn.Linear
    layer_norm0: eqx.nn.LayerNorm
    layer_norm1: eqx.nn.LayerNorm

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, key: jax.random.PRNGKey, layernorm: bool, dropout_p: float = 0.0):
        key, subkey = jax.random.split(key)
        self.multiheaded_attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=dim_in,
            key_size=dim_in,
            qk_size=dim_in,
            vo_size=dim_in,
            value_size=dim_in,
            output_size=dim_out,
            key=subkey,
            dropout_p=dropout_p,
        )
        key, subkey = jax.random.split(key)
        self.output_layer = eqx.nn.Linear(dim_out, dim_out, key=subkey)
        self.layer_norm0 = eqx.nn.LayerNorm(dim_out) if layernorm else None
        self.layer_norm1 = eqx.nn.LayerNorm(dim_out) if layernorm else None

    def __call__(self, x: Array, key=None) -> Array:
        attention_out = self.multiheaded_attention(x, x, x, key=key, inference=True if key is None else False)
        if self.layer_norm0 is not None:
            attention_out = jax.vmap(self.layer_norm0)(attention_out)
        out = attention_out + jax.nn.relu(jax.vmap(self.output_layer)(attention_out))
        if self.layer_norm1 is not None:
            out = jax.vmap(self.layer_norm1)(out)
        return out
