import chex
import equinox as eqx
import jax
import jax.flatten_util
import jax.numpy as jnp
import xarray as xr
from jaxtyping import ArrayLike

from popsim import TimeIndepModule
from popsim.ml import DataLoader
from popsim.modules.linear_autoencoder import LinearAutoEncoder
from popsim.modules.meqml.utils import strip_name_prefix
from popsim.norm_data import apply_norm, norm_data


@chex.dataclass
class Input:
    Ia: ArrayLike  # Active coil currents
    Iu: ArrayLike  # Vessel currents (FGE convention).
    bp: ArrayLike  # Poloidal beta
    rBt: ArrayLike  # Toroidal magnetic field times radius
    Ip: ArrayLike  # Plasma current
    qA: ArrayLike  # Safety factor on axis


@chex.dataclass
class Output:
    Fx: xr.DataArray  # Flux grid.
    Bm: xr.DataArray  # Magnetic probe measurements.
    Ff: xr.DataArray  # Flux loop measurements.


class FGS(TimeIndepModule):
    """ML surrogate for the FGS forward Grad-Shafranov solver. Essentially just LIUQE with fewer inputs"""

    nn: eqx.Module
    Fx_autoencoder: LinearAutoEncoder
    Bm_autoencoder: LinearAutoEncoder
    Ff_autoencoder: LinearAutoEncoder
    means: dict[str, ArrayLike]
    scales: dict[str, ArrayLike]
    Fx_example: xr.DataArray = eqx.field(static=True)
    Bm_example: xr.DataArray = eqx.field(static=True)
    Ff_example: xr.DataArray = eqx.field(static=True)

    def __call__(self, inputs: Input | xr.Dataset) -> Output:
        if isinstance(inputs, xr.Dataset):
            inputs = Input(
                Ia=inputs["LY.Ia"].data,
                Iu=inputs["LY.Iu"].data,
                bp=inputs["LY.bp"].data,
                rBt=inputs["LY.rBt"].data,
                Ip=inputs["LY.Ip"].data,
                qA=inputs["LY.qA"].data,
            )

        inputs_normed = {k: apply_norm(v, self.means[k], self.scales[k]) for k, v in inputs.items()}
        inputs_normed_flat, _ = jax.flatten_util.ravel_pytree(inputs_normed)

        nn_out = self.nn(inputs_normed_flat)

        Fx_out_flat = self.Fx_autoencoder.backward(nn_out)

        Fx_out = xr.DataArray(Fx_out_flat.reshape(self.Fx_example.shape), dims=self.Fx_example.dims, coords=self.Fx_example.coords)

        Bm_out_flat = self.Bm_autoencoder.backward(nn_out)
        Bm_out = xr.DataArray(Bm_out_flat, dims=self.Bm_example.dims, coords=self.Bm_example.coords)

        Ff_out_flat = self.Ff_autoencoder.backward(nn_out)
        Ff_out = xr.DataArray(Ff_out_flat, dims=self.Ff_example.dims, coords=self.Ff_example.coords)

        out = Output(Fx=Fx_out, Bm=Bm_out, Ff=Ff_out)

        return out

    @classmethod
    def init(cls, train_dl: DataLoader, n_latent: int, nn_width: int, nn_depth: int, prng_seed: int):
        input_vars = train_dl.metadata.input_vars

        _, means, scales = norm_data(train_dl.ds, scaling_type="quantile_50", sample_dim="sample")
        means = {strip_name_prefix(k): v.data for k, v in means.data_vars.items()}
        scales = {strip_name_prefix(k): v.data for k, v in scales.data_vars.items()}

        ds = next(iter(train_dl)).ds

        Fx_autoencoder, _ = LinearAutoEncoder.fit(jax.vmap(jnp.ravel)(ds["LY.Fx"].transpose("sample", ...).values), n_latent)
        Bm_autoencoder, _ = LinearAutoEncoder.fit(ds["LY.Bm"].transpose("sample", ...).values, n_latent)
        Ff_autoencoder, _ = LinearAutoEncoder.fit(ds["LY.Ff"].transpose("sample", ...).values, n_latent)

        input_example = ds[input_vars].isel(sample=0)

        flat_input_example = jax.flatten_util.ravel_pytree(input_example)[0]

        nn = eqx.nn.MLP(
            in_size=flat_input_example.size, out_size=n_latent, width_size=nn_width, depth=nn_depth, key=jax.random.PRNGKey(prng_seed)
        )
        ds_example = ds.isel(sample=0).reset_coords(drop=True)

        out = cls(
            nn=nn,
            Fx_autoencoder=Fx_autoencoder,
            Bm_autoencoder=Bm_autoencoder,
            Ff_autoencoder=Ff_autoencoder,
            means=means,
            scales=scales,
            Fx_example=ds_example["LY.Fx"],
            Bm_example=ds_example["LY.Bm"],
            Ff_example=ds_example["LY.Ff"],
        )
        return out
