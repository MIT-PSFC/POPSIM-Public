import chex
import equinox as eqx
import jax
import jax.flatten_util
import xarray as xr

from popsim import TimeIndepModule
from popsim.ml import DataLoader
from popsim.modules.linear_autoencoder import LinearAutoEncoder
from popsim.norm_data import ScalingType


@chex.dataclass
class Inputs:
    Uf: xr.DataArray  # Flux loop voltages.
    Ff: xr.DataArray  # Flux loop time-integrated measurements.
    Bm: xr.DataArray  # Magnetic field measurements.
    Ia: xr.DataArray  # Active coil currents
    Ft: xr.DataArray  # Toroidal flux measurement.
    rBt: xr.DataArray  # Toroidal magnetic field times radius.


@chex.dataclass
class Outputs:
    Fx: xr.DataArray  # Flux grid.
    Ip: xr.DataArray  # Plasma current
    betap: xr.DataArray  # Poloidal beta


class LIUQE(TimeIndepModule):
    diag_encoder: LinearAutoEncoder
    physics_encoder: LinearAutoEncoder
    diag_example: Inputs = eqx.field(static=True)
    physics_example: Outputs = eqx.field(static=True)
    nn: eqx.nn.MLP | None = None  # Optional neural network between encoders.

    def __call__(self, inp: Inputs | xr.Dataset) -> Outputs:
        if isinstance(inp, xr.Dataset):
            diag = Inputs(
                Uf=inp["LX.Uf"].data,
                Ff=inp["LX.Ff"].data,
                Bm=inp["LX.Bm"].data,
                Ia=inp["LX.Ia"].data,
                Ft=inp["LX.Ft"].data,
                rBt=inp["LX.rBt"].data,
            )
        encoded = self.diag_encoder.forward(jax.flatten_util.ravel_pytree(diag)[0])

        if self.nn is not None:
            encoded = self.nn.forward(encoded)

        decoded = self.physics_encoder.backward(encoded)

        physics_unflatten_fn = jax.flatten_util.ravel_pytree(self.physics_example)[1]

        decoded = physics_unflatten_fn(decoded)
        return decoded

    @classmethod
    def init(
        cls,
        dl: DataLoader,
        n_latent: int,
        use_nn: bool,
        nn_depth: int,
        nn_width: int,
        prng_seed: int,
        scaling_type: ScalingType | str = ScalingType.STD,
    ):
        # Use a single batch to initialize the encoders.
        ds = next(iter(dl)).ds.transpose("sample", ...)

        diags = Inputs(
            Uf=ds["LX.Uf"],
            Ff=ds["LX.Ff"],
            Bm=ds["LX.Bm"],
            Ia=ds["LX.Ia"],
            Ft=ds["LX.Ft"],
            rBt=ds["LX.rBt"],
        )

        physics = Outputs(
            Fx=ds["LY.Fx"],
            Ip=ds["LY.Ip"],
            betap=ds["LY.bp"],
        )

        diag_example = jax.tree.map(
            lambda x: x.isel(sample=0).reset_coords(drop=True), diags, is_leaf=lambda x: isinstance(x, xr.DataArray)
        )
        physics_example = jax.tree.map(
            lambda x: x.isel(sample=0).reset_coords(drop=True), physics, is_leaf=lambda x: isinstance(x, xr.DataArray)
        )

        diags = jax.tree.map(lambda x: x.data, diags, is_leaf=lambda x: isinstance(x, xr.DataArray))
        physics = jax.tree.map(lambda x: x.data, physics, is_leaf=lambda x: isinstance(x, xr.DataArray))

        diag_encoder, _ = LinearAutoEncoder.fit_flattened_tree(diags, n_latent, scaling_type)
        physics_encoder, _ = LinearAutoEncoder.fit_flattened_tree(physics, n_latent, scaling_type)

        if use_nn:
            nn = eqx.nn.MLP(
                in_size=n_latent,
                out_size=n_latent,
                width=nn_width,
                depth=nn_depth,
                activation=jax.nn.relu,
                key=jax.random.PRNGKey(prng_seed),
            )
        else:
            nn = None

        return cls(diag_encoder, physics_encoder, diag_example=diag_example, physics_example=physics_example, nn=nn)
