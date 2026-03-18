import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from popsim import TimeDepModule
from popsim.math_utils import soft_clip
from popsim.ml.envs import ModuleTrainingEnv
from popsim.simulate import StepperType

MIN_TAUE = 0.001  # Default inimum reasonable value for tau_e [s]
MAX_TAUE = 0.15  # Maximum reasonable value for tau_e [s]


@chex.dataclass
class TauePredictorInputs:
    # The predictor is based off scaling laws, so we mirror the inputs used in H98 and H89 https://wiki.fusion.ciemat.es/wiki/Scaling_law
    Ip_MA: float  # [MA]
    B0: float  # On axis magnetic field [T]
    ne19: float  # Electron density [10^19 m^-3]
    P_abs_MW: float  # Absorbed power [MW]
    R0: float  # Major radius [m]
    kappa: float  # Elongation [-]
    epsilon: float  # Inverse aspect ratio [-]

    # Including triangularity for the nn
    delta_top: float  # Upper triangularity [-]
    delta_bottom: float  # Lower triangularity [-]


@chex.dataclass
class TauePredictorOutputs:
    taue_pred: float  # [s]
    debug_info: dict | None = None


class BoundedNNPredictor(eqx.Module):
    nn: eqx.Module
    min_val: float = eqx.field(static=True)
    max_val: float = eqx.field(static=True)

    def __call__(self, inp: TauePredictorInputs) -> TauePredictorOutputs:
        arr = jnp.array([inp.Ip_MA, inp.B0, inp.ne19, inp.P_abs_MW, inp.R0, inp.kappa, inp.epsilon, inp.delta_top, inp.delta_bottom])
        nn_out = self.nn(arr)
        bounded_out = soft_clip(nn_out, self.min_val, self.max_val, sharpness=2).squeeze()

        output = TauePredictorOutputs(
            taue_pred=bounded_out,
            debug_info={
                "nn_out": nn_out.squeeze(),  # Squeeze to match dimensions with taue_pred
            },
        )

        return output

    @classmethod
    def create_default(cls):
        return cls(
            nn=eqx.nn.MLP(in_size=9, out_size=1, width_size=32, depth=2, key=jax.random.PRNGKey(42)),
            min_val=MIN_TAUE,
            max_val=MAX_TAUE,
        )


class ScalingLawPredictor(eqx.Module):
    coeff: float
    powers: dict[str, float]
    min_taue: float = eqx.field(static=True)
    max_taue: float = eqx.field(static=True)

    def __init__(self, coeff: float, powers: dict[str, float], min_taue: float = MIN_TAUE, max_taue: float = MAX_TAUE):
        self.coeff = coeff
        self.powers = powers
        self.min_taue = min_taue
        self.max_taue = max_taue

    def __call__(self, inp: TauePredictorInputs) -> TauePredictorOutputs:
        # Ensure each of the input values is strictly greater than 0.01 to avoid numerical instability.
        inp.Ip_MA = jnp.clip(inp.Ip_MA, 0.01, None)
        inp.B0 = jnp.clip(inp.B0, 0.01, None)
        inp.ne19 = jnp.clip(inp.ne19, 0.01, None)
        inp.P_abs_MW = jnp.clip(inp.P_abs_MW, 0.01, None)
        inp.kappa = jnp.clip(inp.kappa, 0.01, None)
        inp.epsilon = jnp.clip(inp.epsilon, 0.01, None)

        scaling_law = self.coeff * (
            inp.Ip_MA ** self.powers["alpha_I"]
            * inp.B0 ** self.powers["alpha_B"]
            * inp.ne19 ** self.powers["alpha_N"]
            * inp.P_abs_MW ** self.powers["alpha_P"]
            * inp.R0 ** self.powers["alpha_R"]
            * inp.kappa ** self.powers["alpha_kappa"]
            * inp.epsilon ** self.powers["alpha_epsilon"]
        )

        # Softmax output
        bounded = soft_clip(scaling_law, self.min_taue, self.max_taue, sharpness=2)
        taue_pred = bounded.squeeze()

        out = TauePredictorOutputs(
            taue_pred=taue_pred,
            debug_info={
                "scaling_law": scaling_law,
            },
        )

        return out

    def get_coeffs_and_powers(self):
        coeffs_and_powers = {
            "coeff": self.coeff,
            "alpha_I": self.powers["alpha_I"],
            "alpha_B": self.powers["alpha_B"],
            "alpha_N": self.powers["alpha_N"],
            "alpha_P": self.powers["alpha_P"],
            "alpha_R": self.powers["alpha_R"],
            "alpha_kappa": self.powers["alpha_kappa"],
            "alpha_epsilon": self.powers["alpha_epsilon"],
        }
        return coeffs_and_powers

    @classmethod
    def create_ipb98(cls) -> "ScalingLawPredictor":
        powers = {
            "alpha_I": jnp.array(0.93),
            "alpha_B": jnp.array(0.15),
            "alpha_N": jnp.array(0.41),
            "alpha_P": jnp.array(-0.69),
            "alpha_R": jnp.array(1.97),
            "alpha_kappa": jnp.array(0.78),
            "alpha_epsilon": jnp.array(0.58),
        }
        return cls(
            coeff=jnp.array(0.0562),
            powers=powers,
        )


class PowerBalance(TimeDepModule):
    taue_predictor: eqx.Module

    @chex.dataclass
    class State:
        Wtot_MJ: float

    @chex.dataclass
    class Inputs:
        taue_predictor_inputs: TauePredictorInputs
        P_rad_MW: float  # Radiated power [MW]

    @chex.dataclass
    class Output:
        Wtot_MJ_pred: float
        P_cond_MW: float
        taue_predictor_output: TauePredictorOutputs

    def __call__(self, state: "State", inputs: "Inputs") -> tuple[State, Output]:
        taue_predictor_output = self.taue_predictor(inputs.taue_predictor_inputs)

        taue_pred = taue_predictor_output.taue_pred

        P_cond_MW = state.Wtot_MJ / taue_pred
        P_rad_MW = inputs.P_rad_MW
        P_abs_MW = inputs.taue_predictor_inputs.P_abs_MW

        Wtot_MJ_dot = P_abs_MW - P_cond_MW - P_rad_MW

        state_dot = PowerBalance.State(Wtot_MJ=Wtot_MJ_dot)
        outputs = PowerBalance.Output(Wtot_MJ_pred=state.Wtot_MJ, P_cond_MW=P_cond_MW, taue_predictor_output=taue_predictor_output)
        return state_dot, outputs

    @staticmethod
    def init(config: dict) -> "PowerBalance":
        if config["model_type"] == "scaling_law":
            taue_predictor = ScalingLawPredictor.create_ipb98()
        elif config["model_type"] == "neural_network":
            taue_predictor = BoundedNNPredictor(
                nn=eqx.nn.MLP(
                    in_size=len(config["network_vars"]),
                    out_size=1,
                    width_size=config["nn_width"],
                    depth=config["nn_depth"],
                    key=jax.random.PRNGKey(config["prng_seed"]),
                ),
                min_val=config["min_val"],
                max_val=config["max_val"],
            )
        else:
            raise ValueError(f"Unknown model type: {config['model_type']}. Must be one of ['scaling_law', 'neural_network'].")

        module = PowerBalance(
            taue_predictor=taue_predictor,
        )
        return module


class PowerBalanceEnv(ModuleTrainingEnv):
    module: PowerBalance
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    @staticmethod
    def create_state(observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]):
        return PowerBalance.State(Wtot_MJ=observations["Wtot_MJ"].data)

    @staticmethod
    def create_inputs(inputs: dict[str, ArrayLike]):
        return PowerBalance.Inputs(
            taue_predictor_inputs=TauePredictorInputs(
                Ip_MA=inputs["Ip_MA"].data,
                B0=inputs["B0"].data,
                ne19=inputs["ne19"].data,
                P_abs_MW=inputs["P_abs_MW"].data,
                R0=inputs["R0"].data,
                kappa=inputs["kappa"].data,
                epsilon=inputs["epsilon"].data,
                delta_top=inputs["delta_top"].data,
                delta_bottom=inputs["delta_bottom"].data,
            ),
            P_rad_MW=inputs["P_rad_MW"].data,
        )

    def get_trainable(self):
        if isinstance(self.module.taue_predictor, ScalingLawPredictor):
            # Specify that only a subset of parameters in the scaling law is trainable.
            scaling_law = self.module.taue_predictor
            return (
                scaling_law.coeff,
                scaling_law.powers["alpha_I"],
                scaling_law.powers["alpha_N"],
                scaling_law.powers["alpha_P"],
                scaling_law.powers["alpha_kappa"],
                scaling_law.powers["alpha_epsilon"],
            )
        elif isinstance(self.module.taue_predictor, BoundedNNPredictor):
            # Specify that all parameters in the NN is trainable.
            nn = self.module.taue_predictor
            return nn
