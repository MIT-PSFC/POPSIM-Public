import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import ArrayLike

from popsim.math_utils import soft_clip
from popsim.module_base import TimeIndepModule


class RadiatedPower(TimeIndepModule):
    """Model that predicts radiated power from plasma parameters.
    Essentially ne * network, bounded within some min/max values."""

    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    @chex.dataclass
    class Inputs:
        # P_RAD depends on impurities, density, and temperature
        # Can infer temperature from stored energy, density, and shaping
        # But we don't have a good handle on impurities so... ignore that for now
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
        P_rad_MW_pred: float
        debug_info: dict

    def __call__(self, inputs: Inputs) -> Output:
        if isinstance(inputs, xr.Dataset):
            inputs = RadiatedPower.Inputs(
                Ip_MA=inputs["Ip_MA"].data,
                R0=inputs["R0"].data,
                a_minor=inputs["a_minor"].data,
                kappa=inputs["kappa"].data,
                delta_top=inputs["delta_top"].data,
                delta_bottom=inputs["delta_bottom"].data,
                ne20=inputs["ne20"].data,
                Wtot_MJ=inputs["Wtot_MJ"].data,
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
        bounded_out = soft_clip(inputs.ne20 * nn_out, self.min_val, self.max_val, sharpness=2).squeeze()

        out = RadiatedPower.Output(
            P_rad_MW_pred=bounded_out,
            debug_info={
                "nn_out": nn_out.squeeze(),  # Squeeze to match dimensions with P_rad_MW_pred
            },
        )

        return out

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
    ) -> "RadiatedPower":
        nn = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=nn_width,
            depth=nn_depth,
            key=jax.random.PRNGKey(prng_seed),
        )
        return cls(nn=nn, min_val=min_val, max_val=max_val)


class RadiatedPowerEnv(eqx.Module):
    module: RadiatedPower

    def __init__(self, module: RadiatedPower):
        self.module = module

    @staticmethod
    def create_inputs(inputs: dict[str, ArrayLike]):
        inputs = RadiatedPower.Inputs(
            plasma_current=inputs["Ip_MA"],
            ne20=inputs["ne20"],
            stored_energy=inputs["Wtot_MJ"],
            epsilon=inputs["epsilon"],
            kappa=inputs["kappa"],
            delta_top=inputs["DELTA_TOP"],
            delta_bottom=inputs["DELTA_BOTTOM"],
        )
        return inputs

    def __call__(self, inputs):
        inputs = self.create_inputs(inputs)
        return self.module(inputs)

    def get_trainable(self):
        return self.module.nn
