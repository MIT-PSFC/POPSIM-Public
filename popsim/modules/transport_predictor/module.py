import chex
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from popsim import TimeDepModule
from popsim.ml.checkpointing import create_default_checkpoint_manager, restore_model
from popsim.ml.envs import ModuleEvalEnvInput, ModuleTrainingEnv, call_module_eval_env
from popsim.ml.train_config import load_dict
from popsim.modules.power_balance.module import PowerBalance, PowerBalanceEnv, TauePredictorInputs
from popsim.modules.power_balance.p_oh.module import OhmicPower
from popsim.modules.power_balance.p_rad.module import RadiatedPower
from popsim.modules.profile_predictor.module import Inputs as ProfilePredictorInputs
from popsim.modules.profile_predictor.module import Outputs as ProfilePredictorOutputs
from popsim.modules.profile_predictor.module import ProfilePredictor
from popsim.simulate import StepperType


class TransportPredictor(TimeDepModule):
    """Predict profiles in a time-dependent manner.
    Uses time-independent ProfilePredictor, OhmicPower, RadiatedPower,
    and a time-dependent PowerBalance module.
    """

    profile_predictor: ProfilePredictor
    power_balance: PowerBalance
    p_oh_predictor: OhmicPower
    p_rad_predictor: RadiatedPower
    rhogrid: tuple = eqx.field(static=True)

    @chex.dataclass
    class State:
        power_balance_state: "PowerBalance.State"

    @chex.dataclass
    class Inputs:
        R0: float
        B0: float
        Ip_MA: float
        a_minor: float
        kappa: float
        delta_top: float
        delta_bottom: float
        P_aux_MW: float
        ne20: float

    @chex.dataclass
    class Output:
        profile_predictor_output: ProfilePredictorOutputs
        power_balance_output: "PowerBalance.Output"
        p_oh_output: "OhmicPower.Output"
        p_rad_output: "RadiatedPower.Output"
        rho: Array

    def __init__(
        self,
        profile_predictor: ProfilePredictor,
        power_balance: PowerBalance,
        p_oh_predictor: OhmicPower,
        p_rad_predictor: RadiatedPower,
        rhogrid: tuple,
    ):
        self.profile_predictor = profile_predictor
        self.power_balance = power_balance
        self.p_oh_predictor = p_oh_predictor
        self.p_rad_predictor = p_rad_predictor
        self.rhogrid = rhogrid

    def __call__(self, state: "State", inputs: Inputs) -> tuple[State, Output]:
        p_oh_predictor_inputs = OhmicPower.Inputs(
            Ip_MA=inputs.Ip_MA,
            R0=inputs.R0,
            a_minor=inputs.a_minor,
            kappa=inputs.kappa,
            delta_top=inputs.delta_top,
            delta_bottom=inputs.delta_bottom,
            ne20=inputs.ne20,
            Wtot_MJ=state.power_balance_state.Wtot_MJ,
        )
        p_oh_predictor_output = self.p_oh_predictor(p_oh_predictor_inputs)
        P_oh_MW_pred = p_oh_predictor_output.P_oh_MW_pred

        p_rad_predictor_inputs = RadiatedPower.Inputs(
            Ip_MA=inputs.Ip_MA,
            R0=inputs.R0,
            a_minor=inputs.a_minor,
            kappa=inputs.kappa,
            delta_top=inputs.delta_top,
            delta_bottom=inputs.delta_bottom,
            ne20=inputs.ne20,
            Wtot_MJ=state.power_balance_state.Wtot_MJ,
        )
        p_rad_predictor_output = self.p_rad_predictor(p_rad_predictor_inputs)
        P_rad_MW_pred = p_rad_predictor_output.P_rad_MW_pred

        P_abs_MW = P_oh_MW_pred + inputs.P_aux_MW

        power_balance_inputs = PowerBalance.Inputs(
            taue_predictor_inputs=TauePredictorInputs(
                Ip_MA=inputs.Ip_MA,
                B0=inputs.B0,
                ne19=inputs.ne20 * 10,
                P_abs_MW=P_abs_MW,
                R0=inputs.R0,
                kappa=inputs.kappa,
                epsilon=inputs.a_minor / inputs.R0,
                delta_top=inputs.delta_top,
                delta_bottom=inputs.delta_bottom,
            ),
            P_rad_MW=P_rad_MW_pred,
        )
        power_balance_state_dot, power_balance_output = self.power_balance(
            state=state.power_balance_state,
            inputs=power_balance_inputs,
        )

        delta = (inputs.delta_top + inputs.delta_bottom) / 2
        profile_predictor_inputs = ProfilePredictorInputs(
            R0=inputs.R0,
            B0=inputs.B0,
            Ip=inputs.Ip_MA,
            a_minor=inputs.a_minor,
            kappa=inputs.kappa,
            delta=delta,
            Paux=inputs.P_aux_MW,
            ne20_line_avg=inputs.ne20,
            Wtot_MJ=state.power_balance_state.Wtot_MJ,
            rho=jnp.array(self.rhogrid),
        )
        profile_predictor_output = self.profile_predictor(profile_predictor_inputs)

        profile_predictor_output = ProfilePredictorOutputs(
            ne=profile_predictor_output.ne.data,
            te=profile_predictor_output.te.data,
        )

        outputs = TransportPredictor.Output(
            profile_predictor_output=profile_predictor_output,
            power_balance_output=power_balance_output,
            p_oh_output=p_oh_predictor_output,
            p_rad_output=p_rad_predictor_output,
            rho=jnp.array(self.rhogrid),
        )
        state_dot = TransportPredictor.State(power_balance_state=power_balance_state_dot)

        return state_dot, outputs

    @classmethod
    def init(
        cls,
        profile_predictor_config: dict,
        power_balance_config: dict,
        p_oh_predictor_config: dict,
        p_rad_predictor_config: dict,
        rhogrid: Array,
        restore_submodules: bool = False,
    ) -> "TransportPredictor":
        profile_predictor_config = load_dict(profile_predictor_config)
        profile_predictor_model_init = profile_predictor_config["model_init_config"]
        profile_predictor = ProfilePredictor.init(
            n_shapes=profile_predictor_model_init["n_shapes"],
            rhogrid=rhogrid,
            nn_width=profile_predictor_model_init["nn_width"],
            nn_depth=profile_predictor_model_init["nn_depth"],
            shape_type=profile_predictor_model_init["shape_type"],
            softmax_temp=profile_predictor_model_init["softmax_temp"],
            use_ne_edge=profile_predictor_model_init.get("use_ne_edge", False),
            prng_seed=profile_predictor_model_init["prng_seed"],
        )
        power_balance_config = load_dict(power_balance_config)
        power_balance_model_init = power_balance_config["model_init_config"]
        power_balance = PowerBalance.init(
            power_balance_model_init,
        )
        p_oh_predictor_config = load_dict(p_oh_predictor_config)
        p_oh_model_init = p_oh_predictor_config["model_init_config"]
        p_oh_predictor = OhmicPower.init(
            **p_oh_model_init,
        )
        p_rad_predictor_config = load_dict(p_rad_predictor_config)
        p_rad_model_init = p_rad_predictor_config["model_init_config"]
        p_rad_predictor = RadiatedPower.init(
            **p_rad_model_init,
        )

        if restore_submodules:
            for config in [
                profile_predictor_config,
                power_balance_config,
                p_oh_predictor_config,
                p_rad_predictor_config,
            ]:
                if not config["checkpoint_dir"]:
                    raise ValueError(f"Checkpoint dir for {config.project} is not provided in the config.")

            profile_predictor_manager = create_default_checkpoint_manager(profile_predictor_config["checkpoint_dir"])
            p_oh_manager = create_default_checkpoint_manager(p_oh_predictor_config["checkpoint_dir"])
            p_rad_manager = create_default_checkpoint_manager(p_rad_predictor_config["checkpoint_dir"])
            power_balance_manager = create_default_checkpoint_manager(power_balance_config["checkpoint_dir"])

            profile_predictor = restore_model(profile_predictor_manager, profile_predictor)
            p_oh_predictor = restore_model(p_oh_manager, p_oh_predictor)
            p_rad_predictor = restore_model(p_rad_manager, p_rad_predictor)

            # Power balance is time-dependent so the checkpoint saved a ModuleTrainingEnv. Load that first and extract the module.
            power_balance_env = PowerBalanceEnv(module=power_balance)
            power_balance_env = restore_model(power_balance_manager, power_balance_env)
            power_balance = power_balance_env.module

        return cls(
            profile_predictor=profile_predictor,
            power_balance=power_balance,
            p_oh_predictor=p_oh_predictor,
            p_rad_predictor=p_rad_predictor,
            rhogrid=tuple(rhogrid.tolist()),
        )


class TransportPredictorEnv(ModuleTrainingEnv):
    module: TransportPredictor
    stepper: StepperType = eqx.field(static=True, default=StepperType.SIMPLE_EULER)

    @staticmethod
    def create_state(observations: dict[str, ArrayLike], inputs: dict[str, ArrayLike]):
        state = TransportPredictor.State(
            power_balance_state=PowerBalance.State(
                Wtot_MJ=observations["Wtot_MJ"].data,
            )
        )
        return state

    @staticmethod
    def create_inputs(inputs: dict[str, ArrayLike]):
        inputs = TransportPredictor.Inputs(
            R0=inputs["R0"].data,
            B0=inputs["B0"].data,
            Ip_MA=inputs["Ip_MA"].data,
            a_minor=inputs["a_minor"].data,
            kappa=inputs["kappa"].data,
            delta_top=inputs["delta_top"].data,
            delta_bottom=inputs["delta_bottom"].data,
            P_aux_MW=inputs["P_aux_MW"].data,
            ne20=inputs["ne20"].data,
        )
        return inputs

    def get_trainable(self):
        # Force the profile predictor shapes to be frozen

        # Get all leaves that are not a part of te_shapes and ne_shapes.
        # All of these leaves are trainable.
        ids_of_shape_leaves = [
            id(x) for x in jax.tree.leaves((self.module.profile_predictor.te_shapes, self.module.profile_predictor.ne_shapes))
        ]
        profile_predictor_leaves = [x for x in jax.tree.leaves(self.module.profile_predictor) if id(x) not in ids_of_shape_leaves]

        return (
            self.module.power_balance,
            self.module.p_oh_predictor,
            self.module.p_rad_predictor,
            profile_predictor_leaves,
        )

    @eqx.filter_jit
    def __call__(self, env_input: ModuleEvalEnvInput) -> diffrax.Solution:
        return call_module_eval_env(self, env_input)
