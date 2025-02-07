from jaxtyping import PyTree
from popsim.tests.fixtures import oscillator_dataset
from popsim.ml.trainer import Trainer
from popsim.ml.dataloading import make_time_dep_dataloader
from popsim.ml.envs import ModuleTrainingEnv
from popsim.ml.loss import IntegralLoss
from popsim.ml.partition import make_partition_by_members
from popsim.ml.split_utils import split_dataset_by_fracs
from popsim import ModuleBase
import chex
import equinox as eqx
import jax.nn as jnn
import jax
import jax.numpy as jnp
import optax
import pytest
import os

@chex.dataclass
class NeuralODE(ModuleBase):
    @chex.dataclass
    class Config:
        nn: eqx.Module
    
    @chex.dataclass
    class State:
        state: dict[str, float]
    
    @chex.dataclass
    class Inputs:
        pass

    @chex.dataclass
    class Output:
        state: dict[str, float]

    config: Config

    def __init__(self, config):
        self.config = config
    
    def __call__(self, state: State, inputs: Inputs) -> tuple[State, Output]:
        state_flat = jnp.asarray(jax.tree.leaves(state.state))
        state_dot_flat = self.config.nn(state_flat)
        state_dot = NeuralODE.State(state=dict(zip(state.state.keys(), state_dot_flat)))
        output = NeuralODE.Output(state=state.state)
        return state_dot, output

class NeuralODEEnv(ModuleTrainingEnv):
    module: NeuralODE
    @staticmethod
    def create_state(data):
        return NeuralODE.State(state={"y0": data["y0"], "y1": data["y1"]})
    
    @staticmethod
    def create_inputs(data):
        return NeuralODE.Inputs()
    
    def get_trainable(self):
        return self.module.config.nn

@pytest.mark.parametrize("use_val", [True, False])
@pytest.mark.parametrize("train_seg_length", [None, 50])
@pytest.mark.parametrize("optimizer", [optax.adabelief(5e-3), optax.lbfgs()])
@pytest.mark.parametrize("batch_size", [None, 1, 8])
def test_train_neural_ode(oscillator_dataset, use_val, train_seg_length, optimizer, batch_size, tmpdir):
    ds = oscillator_dataset

    nn = eqx.nn.MLP(in_size=2, out_size=2, width_size=64, depth=2, activation=jnn.softplus, key=jax.random.PRNGKey(0))
    module = NeuralODE(config=NeuralODE.Config(nn=nn))
    env = NeuralODEEnv(module=module)

    def loss(predictions, targets):
        y0_loss = optax.losses.l2_loss(predictions.state["y0"], targets["y0"])
        y1_loss = optax.losses.l2_loss(predictions.state["y1"], targets["y1"])
        return y0_loss + y1_loss


    if use_val:
        ds, val_ds = split_dataset_by_fracs(ds, (0.8, 0.2), "simulation", 42)
        val_dl = make_time_dep_dataloader(
            val_ds,
            time_coord="time",
            episode_coord="simulation",
            state_init_vars=["y0", "y1"],
            input_vars=[],
            target_vars=["y0", "y1"],
            segment_length=None,
            batch_size=batch_size
        )
    else:
        val_dl = None

    dl = make_time_dep_dataloader(
        ds,
        time_coord="time",
        episode_coord="simulation",
        state_init_vars=["y0", "y1"],
        input_vars=[],
        target_vars=["y0", "y1"],
        segment_length=train_seg_length,
        batch_size=batch_size
    )

    trainer = Trainer(
        model=env,
        loss_fn=IntegralLoss(loss),
        optimizer=optimizer,
        checkpoint_dir=tmpdir,
    )

    loss_start = trainer.compute_loss(dl if not use_val else val_dl)
    trainer.train(
        train_dl=dl,
        val_dl=val_dl,
        # Only val once to help ensure that the last epoch is the best, and hence the checkpoint is saved.
        max_epochs=50,
        epochs_per_val=50,
    )
    loss_end = trainer.compute_loss(dl if not use_val else val_dl)

    if batch_size == 1:
        # The batch size 1 case can be highly unstable, so expect it to not train well.
        assert loss_end["mean"] != loss_start["mean"]
        return

    assert loss_end["mean"]/loss_start["mean"] < 0.5


    if use_val:
        # Checkpoints are only saved when using a validation set.
        # Check that the checkpoint directory only has one file.
        assert len(os.listdir(tmpdir)) == 1


        # Create a new trainer and load the checkpoint.
        new_trainer = Trainer(
            model=env,
            loss_fn=IntegralLoss(loss),
            optimizer=optimizer,
            checkpoint_dir=tmpdir,
        )

        # Check that the train_state of the new_trainer is not the same as the old trainer.
        assert new_trainer.train_state.step == 0
        assert new_trainer.train_state.epoch == 0

        # Check that the models are not the same.
        equals_tree = jax.tree.map(lambda x, y: jnp.all(x == y), trainer.train_state.model, new_trainer.train_state.model)
        equals_tree_leaves = jax.tree.leaves(equals_tree)
        # Check that the leaves of the equals_tree are not all True.
        assert not all(equals_tree_leaves)

        # Now restore the best checkpoint with the new_trainer.
        new_trainer.restore_best_checkpoint()
        assert new_trainer.train_state.step > 0
        assert new_trainer.train_state.epoch > 0

        # Check that the restored new_trainer is the same as the old trainer at the end of training.
        chex.assert_trees_all_equal(trainer.train_state, new_trainer.train_state)