import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array

from popsim import TimeIndepModule
from popsim.ml.rtd_mlp import Activation, RtdMLP
from popsim.modules.meqml.utils import strip_name_prefix
from popsim.modules.set_attention import SetAttentionBlock
from popsim.norm_data import apply_norm, apply_unnorm, norm_data_xr


@chex.dataclass
class Input:
    rc: xr.DataArray
    zc: xr.DataArray
    lcC: xr.DataArray
    lcS: xr.DataArray
    lcX1: xr.DataArray
    lcX2: xr.DataArray
    lcD: xr.DataArray
    lcI: xr.DataArray
    Ip: xr.DataArray  # Plasma current
    bp: xr.DataArray  # Poloidal beta
    qA: xr.DataArray  # Safety factor on axis
    rBt: xr.DataArray  # Toroidal magnetic field times radius
    rl: xr.DataArray  # R-position of limiter points.
    zl: xr.DataArray  # Z-position of limiter points.


@chex.dataclass
class Output:
    Ia: xr.Variable  # Active coil currents
    debug: dict | None = None


class SetLinear(TimeIndepModule):
    linear: eqx.nn.Linear

    def __init__(self, in_size: int, out_size: int, key: jax.random.PRNGKey):
        self.linear = eqx.nn.Linear(in_size, out_size, key=key)

    def __call__(self, x: jax.Array, key=None) -> jax.Array:
        return jax.vmap(self.linear)(x)


class FBTSurrogate(TimeIndepModule):
    mlp: RtdMLP
    means: dict[str, Array]
    scales: dict[str, Array]
    cp_encoder: SetAttentionBlock | SetLinear
    limiter_encoder: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    attention_input_scale: jax.Array

    def __call__(self, inputs: Input | xr.Dataset, debug: bool = False) -> Output:
        if isinstance(inputs, xr.Dataset):
            input_dict = {strip_name_prefix(k): inputs[k].data for k in inputs.data_vars}
            inputs = Input(
                rc=input_dict["rc"],
                zc=input_dict["zc"],
                lcC=input_dict["lcC"],
                lcS=input_dict["lcS"],
                lcX1=input_dict["lcX1"],
                lcX2=input_dict["lcX2"],
                lcD=input_dict["lcD"],
                lcI=input_dict["lcI"],
                Ip=input_dict["Ip"],
                bp=input_dict["bp"],
                qA=input_dict["qA"],
                rBt=input_dict["rBt"],
                rl=input_dict["rl"],
                zl=input_dict["zl"],
            )
        prng_key = jax.random.PRNGKey(input_dict["prng_key"]) if "prng_key" in input_dict else None

        def pred(inp):
            # Normalize inputs, but not the points.
            # Don't bother normalizing the points since they're already all order 1.
            inp = dict(inp.items())
            inp["Ip"] = apply_norm(inp["Ip"], self.means["LY.Ip"], self.scales["LY.Ip"])
            inp["bp"] = apply_norm(inp["bp"], self.means["LY.bp"], self.scales["LY.bp"])
            inp["qA"] = apply_norm(inp["qA"], self.means["LY.qA"], self.scales["LY.qA"])
            inp["rBt"] = apply_norm(inp["rBt"], self.means["LY.rBt"], self.scales["LY.rBt"])

            cp_encoder_input = jnp.stack(
                [inp["rc"], inp["zc"], inp["lcC"], inp["lcS"], inp["lcX1"], inp["lcX2"], inp["lcD"], inp["lcI"]], axis=1
            )

            cp_encoder_input = self.attention_input_scale * cp_encoder_input
            cp_latent = self.cp_encoder(cp_encoder_input, key=prng_key)
            if prng_key is not None:
                cp_latent = self.dropout(cp_latent, key=prng_key + 1)

            cp_latent = jnp.mean(cp_latent, axis=0)

            limiter_latent = self.limiter_encoder(jnp.concatenate([inp["rl"], inp["zl"]])).squeeze()

            latent = jnp.concatenate(
                [
                    cp_latent,
                    jnp.array([inp["Ip"], inp["bp"], inp["qA"], inp["rBt"], limiter_latent]),
                ],
            )
            ia_pred_norm = self.mlp(latent)
            Ia_pred = apply_unnorm(ia_pred_norm, self.means["LY.Ia"], self.scales["LY.Ia"])
            return Ia_pred

        Ia_pred = pred(inputs)

        output = Output(
            Ia=xr.Variable(dims=["active_coils"], data=Ia_pred),
            debug=None,
        )

        return output

    @classmethod
    def init(
        cls,
        train_dl,
        nn_width: int,
        nn_depth: int,
        n_latent: int,
        n_heads: int,
        layernorm: bool,
        attention_input_scale: float,
        activation: Activation | str,
        dropout_rate: float,
        prng_seed: int,
    ):
        ds = next(iter(train_dl)).ds
        activation = Activation(activation)

        _, means, scales = norm_data_xr(ds, scaling_type="quantile_50", sample_dim="sample")

        means = {k: v.data for k, v in means.data_vars.items()}
        scales = {k: v.data for k, v in scales.data_vars.items()}

        n_coil = ds.active_coils.size

        nn = RtdMLP(
            in_size=n_latent + 5,
            out_size=n_coil,
            width_size=nn_width,
            depth=nn_depth,
            activation=activation,
            key=jax.random.PRNGKey(prng_seed),
        )

        _, subkey = jax.random.split(jax.random.PRNGKey(prng_seed))
        set_attention = SetAttentionBlock(
            dim_in=8,
            dim_out=n_latent,
            num_heads=n_heads,
            key=subkey,
            layernorm=layernorm,
            dropout_p=dropout_rate,
        )

        dropout = eqx.nn.Dropout(dropout_rate)

        limiter_encoder = eqx.nn.Linear(
            in_features=2 * ds.sizes["oq"],
            out_features=1,
            key=subkey,
        )

        return cls(
            mlp=nn,
            means=means,
            scales=scales,
            cp_encoder=set_attention,
            limiter_encoder=limiter_encoder,
            dropout=dropout,
            attention_input_scale=attention_input_scale,
        )
