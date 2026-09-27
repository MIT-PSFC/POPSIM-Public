import contextlib
import os

import chex
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr

from popsim import TimeDepModule
from popsim.ml.checkpointing import TrainState
from popsim.ml.dataloading import DataLoader, make_time_dep_dataloader, make_time_indep_dataloader
from popsim.ml.envs import ModuleTrainingEnv
from popsim.ml.eval import make_val_loss_eval_fn
from popsim.ml.loggers import LoggerBase, NullLogger
from popsim.ml.loss import IntegralLoss
from popsim.ml.partition import make_partition_by_members
from popsim.ml.split_utils import split_dataset_by_fracs
from popsim.ml.trainer import MAX_HIGH_SKIP_EPOCHS, Trainer, TrainingDivergedError, train_epoch
from popsim.simulate import StepperType
from popsim.tests.fixtures import oscillator_dataset  # noqa: F401  (pytest fixture)


class NeuralODE(TimeDepModule):
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
    def create_state(observations, inputs):
        return NeuralODE.State(state={"y0": observations["y0"], "y1": observations["y1"]})

    @staticmethod
    def create_inputs(inputs):
        return NeuralODE.Inputs()

    def get_trainable(self):
        return self.module.config.nn

@pytest.mark.parametrize("use_val", [True, False])
@pytest.mark.parametrize("train_seg_length", [None, 31]) # 31 is chosen as an unusual segment length to test the code.
@pytest.mark.parametrize("optimizer", [optax.adabelief(5e-3), optax.lbfgs()])
@pytest.mark.parametrize("batch_size", [None, 1, 8])
@pytest.mark.parametrize("stepper", [StepperType.DIFFRAX_EULER, StepperType.SIMPLE_EULER])
def test_train_neural_ode(oscillator_dataset, use_val, train_seg_length, optimizer, batch_size, stepper, tmpdir):
    ds = oscillator_dataset

    nn = eqx.nn.MLP(in_size=2, out_size=2, width_size=16, depth=2, activation=jnn.softplus, key=jax.random.PRNGKey(0))
    module = NeuralODE(config=NeuralODE.Config(nn=nn))
    env = NeuralODEEnv(module=module, stepper=stepper)

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
    # Without a validation dataloader the trainer warns that checkpoints won't be saved.
    warn_ctx = pytest.warns(UserWarning, match="No validation DataLoader") if not use_val else contextlib.nullcontext()
    try:
        with warn_ctx:
            trainer.train(
                train_dl=dl,
                val_dl=val_dl,
                # Only val once to help ensure that the last epoch is the best, and hence the checkpoint is saved.
                max_epochs=20,
                epochs_per_val=20,
            )
    except RuntimeError as e:
        # Encountered a NaN during training
        # This is to be expected for some configurations, especially with batch_size=1 and float32 precision.
        if not (batch_size == 1 and jax.config.jax_enable_x64 is False):
            raise e

    loss_end = trainer.compute_loss(dl if not use_val else val_dl)

    if batch_size == 1:
        # The batch size 1 case can be highly unstable, so expect it to not train well.
        assert loss_end["mean"] != loss_start["mean"]
        return

    assert loss_end["mean"]/loss_start["mean"] < 0.6


    if use_val:
        # Checkpoints are only saved when using a validation set.
        # Validation runs once, on the final epoch, so the best and latest checkpoints are the same single step.
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


class CountingLogger(LoggerBase):
    def __init__(self):
        self.val_epochs = []

    def log(self, dictionary):
        if "val/epoch" in dictionary:
            self.val_epochs.append(dictionary["val/epoch"])


def test_early_stopping_patience(oscillator_dataset):
    """With a zero learning rate the validation loss never improves, so training should stop
    after the first validation establishes the best loss and `patience` further validations pass."""
    ds = oscillator_dataset

    nn = eqx.nn.MLP(in_size=2, out_size=2, width_size=16, depth=2, activation=jnn.softplus, key=jax.random.PRNGKey(0))
    module = NeuralODE(config=NeuralODE.Config(nn=nn))
    env = NeuralODEEnv(module=module, stepper=StepperType.SIMPLE_EULER)

    def loss(predictions, targets):
        y0_loss = optax.losses.l2_loss(predictions.state["y0"], targets["y0"])
        y1_loss = optax.losses.l2_loss(predictions.state["y1"], targets["y1"])
        return y0_loss + y1_loss

    ds, val_ds = split_dataset_by_fracs(ds, (0.8, 0.2), "simulation", 42)
    dl_kwargs = dict(
        time_coord="time",
        episode_coord="simulation",
        state_init_vars=["y0", "y1"],
        input_vars=[],
        target_vars=["y0", "y1"],
        segment_length=None,
        batch_size=None,
        # Keep sample order deterministic so the validation loss is bit-identical between epochs.
        shuffle=False,
    )
    dl = make_time_dep_dataloader(ds, **dl_kwargs)
    val_dl = make_time_dep_dataloader(val_ds, **dl_kwargs)

    trainer = Trainer(
        model=env,
        loss_fn=IntegralLoss(loss),
        optimizer=optax.sgd(learning_rate=0.0),
    )

    patience = 1
    counting_logger = CountingLogger()
    trainer.train(
        train_dl=dl,
        val_dl=val_dl,
        max_epochs=20,
        epochs_per_val=1,
        patience=patience,
        logger=counting_logger,
    )

    # One validation to set the best loss, then `patience` validations with no improvement.
    assert len(counting_logger.val_epochs) == 1 + patience
    assert trainer.train_state.epoch < 20


class _LinearModel(eqx.Module):
    w: jax.Array

    def __call__(self, inputs):
        return {"y": self.w * inputs["x"]}


class _SqrtInputModel(eqx.Module):
    w: jax.Array

    def __call__(self, inputs):
        # A negative input makes the forward pass of that sample alone NaN.
        return {"y": self.w * jnp.sqrt(inputs["x"])}


class _StopRequestingLogger(NullLogger):
    def stop_requested(self) -> bool:
        return True


def _squared_error(prediction, target):
    return jnp.square(prediction["y"] - target["y"])


def _xy_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int | None, pad_last: bool = False) -> DataLoader:
    """An unshuffled time-independent dataloader over one episode with input x and target y."""
    ds = xr.Dataset(
        {"x": (("episode", "time"), x.reshape(1, -1)), "y": (("episode", "time"), y.reshape(1, -1))},
        coords={"episode": [0], "time": np.arange(x.size, dtype=float)},
    )
    return make_time_indep_dataloader(
        ds,
        time_coord="time",
        episode_coord="episode",
        input_vars=["x"],
        target_vars=["y"],
        batch_size=batch_size,
        shuffle=False,
        pad_last=pad_last,
    )


@pytest.mark.parametrize("optimizer", [optax.sgd(0.05), optax.lbfgs()], ids=["sgd", "lbfgs"])
@pytest.mark.parametrize(
    "x_values, batch_size, pad_last, x_values_clean, n_steps_masked",
    [
        # The final batch [5, 6] is padded to [5, 6, 6, 6], which would overweight 6 without the mask.
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4, True, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0),
        # The retry masks out the NaN sample -2.
        ([1.0, -2.0, 3.0, 4.0], None, False, [1.0, 3.0, 4.0], 1),
        # The final batch [5, 6, -7] is padded with a copy of the NaN sample, so the retry must replace it too.
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -7.0], 4, True, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1),
    ],
    ids=["padded", "nan_retry", "padded_nan_retry"],
)
def test_masked_train_epoch_matches_clean_batches(optimizer, x_values, batch_size, pad_last, x_values_clean, n_steps_masked):
    """Padded rows and NaN samples add no loss or gradient.

    An epoch with them masked gives the same train state as an epoch on the same batches without them.
    lbfgs also checks that its line search evaluates the masked loss.
    """
    partition_fn = make_partition_by_members(lambda m: m)

    def _trained_epoch(x_list: list[float], pad_final_batch: bool) -> tuple[TrainState, dict]:
        x = np.asarray(x_list)
        y = np.ones_like(x)
        dl = _xy_dataloader(x, y, batch_size, pad_last=pad_final_batch)
        model = _SqrtInputModel(w=jnp.array(0.5))
        train_state = TrainState.create_new(model, partition_fn, optimizer)
        return train_epoch(train_state, partition_fn, _squared_error, optimizer, dl)

    train_state, train_metrics = _trained_epoch(x_values, pad_last)
    train_state_clean, _ = _trained_epoch(x_values_clean, pad_final_batch=False)

    assert train_metrics["train/nan_steps_masked"] == n_steps_masked
    assert train_metrics["train/nan_steps_skipped"] == 0
    # Not bit-identical, because the masked sum can reduce in a different order than the clean one.
    chex.assert_trees_all_close(train_state, train_state_clean, rtol=1e-12)


def test_diverged_training_falls_back_to_best_checkpoint(tmpdir):
    """Skipping most steps for MAX_HIGH_SKIP_EPOCHS consecutive epochs aborts training.

    With batch_size=1 a NaN sample cannot be masked out, so 3 of the 4 steps are skipped every epoch.
    Without a checkpoint the TrainingDivergedError propagates.
    With one, the test eval runs on the best validated checkpoint from before the divergence.
    """
    x_train = np.array([1.0, -2.0, -3.0, -4.0])
    y_train = np.ones_like(x_train)
    train_dl = _xy_dataloader(x_train, y_train, batch_size=1)
    x_val = np.array([1.0, 2.0])
    y_val = np.ones_like(x_val)
    val_dl = _xy_dataloader(x_val, y_val, batch_size=None)
    test_eval_suite = {"loss": make_val_loss_eval_fn(_squared_error)}
    train_kwargs = dict(
        train_dl=train_dl, val_dl=val_dl, test_dl=val_dl, test_eval_suite=test_eval_suite, max_epochs=10, logger=NullLogger()
    )

    def _make_trainer(checkpoint_dir):
        model = _SqrtInputModel(w=jnp.array(0.5))
        return Trainer(model=model, loss_fn=_squared_error, optimizer=optax.sgd(0.05), checkpoint_dir=checkpoint_dir)

    trainer_without_checkpoint = _make_trainer(checkpoint_dir=None)
    with pytest.raises(TrainingDivergedError):
        trainer_without_checkpoint.train(**train_kwargs)
    assert trainer_without_checkpoint.train_state.epoch == MAX_HIGH_SKIP_EPOCHS

    trainer = _make_trainer(checkpoint_dir=tmpdir)
    test_results = trainer.train(**train_kwargs)
    assert "test/loss" in test_results
    # The val loss improves every epoch, so the best checkpoint is the last one validated before the abort.
    assert trainer.checkpoint_manager.all_steps() == [MAX_HIGH_SKIP_EPOCHS - 1]
    assert trainer.train_state.epoch == MAX_HIGH_SKIP_EPOCHS - 1


def test_stop_request_ends_training_without_test_eval(tmpdir):
    """A logger stop request ends training at the first epoch boundary and skips the test eval."""
    x = np.linspace(0.1, 1.0, 8)
    dl = _xy_dataloader(x, 2.0 * x, batch_size=4)
    test_eval_suite = {"loss": make_val_loss_eval_fn(_squared_error)}
    model = _LinearModel(w=jnp.array(0.0))
    trainer = Trainer(model=model, loss_fn=_squared_error, optimizer=optax.adam(0.05), checkpoint_dir=tmpdir)

    test_results = trainer.train(
        train_dl=dl, val_dl=dl, test_dl=dl, test_eval_suite=test_eval_suite, max_epochs=5, logger=_StopRequestingLogger()
    )
    assert test_results is None
    assert trainer.train_state.epoch == 1


def test_resume_across_wall_budget_matches_uninterrupted(tmpdir):
    """Stopping on the wall budget and resuming twice gives exactly the uninterrupted result.

    Session 1 stops after epoch 1, which has no validation, so it saves a checkpoint without a loss.
    Session 2 resumes there, validates epoch 2 and stops again.
    Session 3 finishes and runs the test eval on the best checkpoint.
    """
    x = np.linspace(0.1, 1.0, 8)
    # No shuffle, so a resumed run sees the same batches as an uninterrupted one.
    dl = _xy_dataloader(x, 2.0 * x, batch_size=4)
    test_eval_suite = {"loss": make_val_loss_eval_fn(_squared_error)}
    train_kwargs = dict(
        train_dl=dl, val_dl=dl, test_dl=dl, test_eval_suite=test_eval_suite, max_epochs=5, epochs_per_val=2, logger=NullLogger()
    )

    def _make_trainer(checkpoint_dir, resume=False):
        model = _LinearModel(w=jnp.array(0.0))
        return Trainer(model=model, loss_fn=_squared_error, optimizer=optax.adam(0.05), checkpoint_dir=checkpoint_dir, resume=resume)

    reference = _make_trainer(tmpdir / "reference")
    reference_test_results = reference.train(**train_kwargs)

    checkpoint_dir = tmpdir / "resumed"
    session_1 = _make_trainer(checkpoint_dir)
    assert session_1.train(**train_kwargs, max_wall_seconds=0.0) is None
    assert session_1.checkpoint_manager.all_steps() == [1]
    assert session_1.checkpoint_manager.best_step() is None

    session_2 = _make_trainer(checkpoint_dir, resume=True)
    assert session_2.train_state.epoch == 1
    assert session_2.train(**train_kwargs, max_wall_seconds=0.0) is None
    assert session_2.checkpoint_manager.all_steps() == [2]
    assert session_2.checkpoint_manager.best_step() == 2

    session_3 = _make_trainer(checkpoint_dir, resume=True)
    assert session_3.train_state.epoch == 2
    resumed_test_results = session_3.train(**train_kwargs)
    assert session_3.checkpoint_manager.all_steps() == [5]

    chex.assert_trees_all_equal(session_3.train_state, reference.train_state)
    chex.assert_trees_all_equal(resumed_test_results, reference_test_results)
