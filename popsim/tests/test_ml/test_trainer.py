from popsim.tests.fixtures import oscillator_dataset
from popsim.ml.trainer import Trainer
from popsim.ml.dataloading import make_dataloader
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.loss import IntegralLoss
from popsim.ml.partition import partition_by_arraylike
from popsim.ml.split_utils import split_dataset_by_coords
from popsim import ModuleBase
import chex
import equinox as eqx
import jax.nn as jnn
import jax
import jax.numpy as jnp
import optax
import pytest

@pytest.mark.parametrize("use_val", [True, False])
@pytest.mark.parametrize("train_seg_length", [None, 50])
def test_train_neural_ode(oscillator_dataset, use_val, train_seg_length):
    ds = oscillator_dataset

    @chex.dataclass
    class NeuralODE(ModuleBase):
        @chex.dataclass
        class Config:
            nn: eqx.Module
        
        @chex.dataclass
        class State:
            state: dict[str, float]
        
        @chex.dataclass
        class Params:
            pass

        @chex.dataclass
        class Output:
            state: dict[str, float]

        config: Config

        def __init__(self, config):
            self.config = config
        
        def __call__(self, state: State, params: Params) -> tuple[State, Output]:
            state_flat = jnp.asarray(jax.tree.leaves(state.state))
            state_dot_flat = self.config.nn(state_flat)
            state_dot = NeuralODE.State(state=dict(zip(state.state.keys(), state_dot_flat)))
            output = NeuralODE.Output(state=state.state)
            return state_dot, output

    class NeuralODEEnv(ModuleEvalEnv):
        @staticmethod
        def create_state(data):
            return NeuralODE.State(state={"y0": data["y0"], "y1": data["y1"]})
        
        @staticmethod
        def create_params(data):
            return NeuralODE.Params()
    
    nn = eqx.nn.MLP(in_size=2, out_size=2, width_size=64, depth=2, activation=jnn.softplus, key=jax.random.PRNGKey(0))
    module = NeuralODE(config=NeuralODE.Config(nn=nn))
    env = NeuralODEEnv(module=module)

    def loss(predictions, targets):
        y0_loss = optax.losses.l2_loss(predictions.state["y0"], targets["y0"])
        y1_loss = optax.losses.l2_loss(predictions.state["y1"], targets["y1"])
        return y0_loss + y1_loss


    if use_val:
        ds, val_ds = split_dataset_by_coords(ds, (0.8, 0.2), "simulation", key=jax.random.PRNGKey(0))
        val_dl = make_dataloader(
            val_ds,
            time_coord="time",
            episode_coord="simulation",
            state_init_vars=["y0", "y1"],
            param_vars=[],
            target_vars=["y0", "y1"],
            segment_length=None
        )
    else:
        val_dl = None


    dl = make_dataloader(
        ds,
        time_coord="time",
        episode_coord="simulation",
        state_init_vars=["y0", "y1"],
        param_vars=[],
        target_vars=["y0", "y1"],
        segment_length=train_seg_length
    )

    trainer = Trainer(
        model=env,
        trainable_params_getter=lambda _env: (_env.module.config.nn),
        loss_fn=IntegralLoss(loss),
        optimizer=optax.adabelief(5e-3),
    )

    loss_start = trainer.compute_loss(dl if not use_val else val_dl)
    trainer.train(
        train_dl=dl,
        val_dl=val_dl,
        max_epochs=50
    )
    loss_end = trainer.compute_loss(dl if not use_val else val_dl)

    assert loss_end/loss_start < 0.5