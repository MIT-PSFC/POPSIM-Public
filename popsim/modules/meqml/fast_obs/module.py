import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import ArrayLike

from popsim import TimeIndepModule
from popsim.ml import DataLoader
from popsim.ml.rtd_mlp import Activation, RtdMLP
from popsim.modules.meqml.utils import strip_name_prefix
from popsim.norm_data import apply_norm, apply_unnorm, norm_data


@chex.dataclass
class Input:
    Ff: xr.DataArray  # Flux loop time-integrated measurements.
    Bm: xr.DataArray  # Magnetic field measurements.
    Ia: xr.DataArray  # Active coil currents


@chex.dataclass
class Output:
    Ip: ArrayLike
    rIp: ArrayLike
    zIp: ArrayLike


class FastObs(TimeIndepModule):
    means: dict[str, ArrayLike]
    scales: dict[str, ArrayLike]
    diag_encoder: eqx.nn.Linear
    nn: eqx.Module
    dropout: eqx.nn.Dropout

    def __call__(self, inputs: Input | xr.Dataset) -> Output:
        if isinstance(inputs, xr.Dataset):
            input_dict = {strip_name_prefix(k): inputs[k].data for k in inputs.data_vars}
            inputs = Input(
                Ff=input_dict["Ff"],
                Bm=input_dict["Bm"],
                Ia=input_dict["Ia"],
            )

        prng_key = jax.random.PRNGKey(input_dict["prng_key"]) if "prng_key" in input_dict else None

        def pred(inp):
            inp_normed = {
                "Ff": apply_norm(inp["Ff"], self.means["LX.Ff"], self.scales["LX.Ff"]),
                "Bm": apply_norm(inp["Bm"], self.means["LX.Bm"], self.scales["LX.Bm"]),
                "Ia": apply_norm(inp["Ia"], self.means["LX.Ia"], self.scales["LX.Ia"]),
            }
            input_normed_flat = jnp.concatenate(
                [
                    inp_normed["Ff"],
                    inp_normed["Bm"],
                    inp_normed["Ia"],
                ]
            )

            encoded = self.diag_encoder(input_normed_flat)

            if prng_key is not None:
                encoded = self.dropout(encoded, key=prng_key)

            output = self.nn(encoded)

            Ip_pred = apply_unnorm(output[0], self.means["LY.Ip"], self.scales["LY.Ip"])
            rIp_pred = apply_unnorm(output[1], self.means["LY.rIp"], self.scales["LY.rIp"])
            zIp_pred = apply_unnorm(output[2], self.means["LY.zIp"], self.scales["LY.zIp"])
            output = Output(
                Ip=Ip_pred,
                rIp=rIp_pred,
                zIp=zIp_pred,
            )
            return output

        output = pred(inputs)

        return output

    @classmethod
    def init(
        cls,
        train_dl: DataLoader,
        nn_width: int,
        nn_depth: int,
        n_latent: int,
        dropout_rate: float,
        activation: Activation | str,
        prng_seed: int,
    ):
        ds = next(iter(train_dl)).ds
        activation = Activation(activation)

        _, means, scales = norm_data(ds, scaling_type="quantile_50", sample_dim="sample")
        means = {k: v.data for k, v in means.data_vars.items()}
        scales = {k: v.data for k, v in scales.data_vars.items()}

        n_active_coils = ds.active_coils.size
        n_mag_probe = ds.mag_probe.size
        n_flux_loop = ds.flux_loop.size

        key = jax.random.PRNGKey(prng_seed)
        key, subkey = jax.random.split(key)
        diag_encoder = eqx.nn.Linear(
            in_features=n_active_coils + n_mag_probe + n_flux_loop,
            out_features=n_latent,
            key=subkey,
        )
        key, subkey = jax.random.split(key)
        nn = RtdMLP(
            in_size=n_latent,
            out_size=3,
            width_size=nn_width,
            depth=nn_depth,
            activation=activation,
            key=subkey,
        )
        dropout = eqx.nn.Dropout(dropout_rate)
        model = cls(
            means=means,
            scales=scales,
            diag_encoder=diag_encoder,
            nn=nn,
            dropout=dropout,
        )
        return model
