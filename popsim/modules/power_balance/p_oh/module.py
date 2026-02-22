import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr

from popsim.math_utils import soft_clip
from popsim.module_base import TimeIndepModule


class OhmicPower(TimeIndepModule):
    """Model that predicts ohmic heating power from plasma parameters.
    Essentially Ip * network, bounded within some min/max values.
    """

    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    @chex.dataclass
    class Inputs:
        # eqn 15.10 of Friedberg
        # P_oh depends on plasma current, temperature, and shaping
        # temperature can be inferred from stored energy, density, and shaping
        Ip_MA: float
        R0: float
        a_minor: float
        kappa: float
        delta_top: float
        delta_bottom: float
        ne20: float
        Wtot_MJ: float

    @chex.dataclass
    class Output:
        P_oh_MW_pred: float
        debug_info: dict

    def __call__(self, inputs: Inputs) -> Output:
        if isinstance(inputs.Ip_MA, xr.DataArray):
            if isinstance(inputs["Wtot_MJ"], xr.DataArray):
                Wtot_MJ = inputs["Wtot_MJ"].data
            else:
                Wtot_MJ = inputs["Wtot_MJ"]
            inputs = OhmicPower.Inputs(
                Ip_MA=inputs["Ip_MA"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bottom=inputs["delta_bottom"].data,
                ne20=inputs["ne20"].data,
                Wtot_MJ=Wtot_MJ,
            )

        arr = jnp.array(
            [
                inputs.Ip_MA,
                inputs.R0,
                inputs.a_minor,
                inputs.kappa,
                inputs.delta_top,
                inputs.delta_bottom,
                inputs.ne20,
                inputs.Wtot_MJ,
            ],
        )
        nn_out = self.nn(arr)
        bounded_out = soft_clip(jnp.abs(inputs.Ip_MA) * nn_out, self.min_val, self.max_val, sharpness=2).squeeze()

        output = OhmicPower.Output(
            P_oh_MW_pred=bounded_out,
            debug_info={
                "nn_out": nn_out.squeeze(),  # Squeeze to match dimensions with P_oh_MW_pred
            },
        )

        return output

    @classmethod
    def init(
        cls,
        in_size: int,
        out_size: int,
        nn_width: int,
        nn_depth: int,
        min_val: float,
        max_val: float,
        prng_seed: int,
    ) -> "OhmicPower":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(nn=nn, min_val=min_val, max_val=max_val)
