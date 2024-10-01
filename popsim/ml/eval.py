from typing import Any, Callable, NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jax_dataloader import DataLoader
from jaxtyping import Array, PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import DEFAULT_SAMPLE_DIM, XarrayPreppedDataset
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.loss import IntegralLoss, LossFunction
from popsim.xarray_utils import DEFAULT_SIM_DIM_NAME, DEFAULT_TIME_DIM_NAME, solution_to_xarray

"""
This module contains utilities for evaluating models on data.
"""


class EvalFnInput(NamedTuple):
    model: TrainableModel  # The model to train.
    dataloader: DataLoader  # DataLoader that was used to evaluate the module.
    output_ds: xr.Dataset  # Output dataset from the module evaluation.

    @property
    def input_ds(self):
        """
        Get the xarray dataset that was used as input to the module evaluation.
        """
        return self.dataloader.dataloader.dataset.ds


# An evaluation function is a function that takes an EvalFnInput and returns a value.
EvaluationFn = Callable[[EvalFnInput], Any]

# An evaluation suite is defined as a dictionary of evaluation functions
EvaluationSuite = dict[str, Callable[[xr.Dataset, xr.Dataset], Any]]


def eval_module_on_data(
    env: ModuleEvalEnv, dataloader: DataLoader, evaluation_suite: Optional[EvaluationSuite] = None
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Evaluate a module given data from a dataloader and an evaluation suite.

    Args:
        env (ModuleEvalEnv): the module wrapped in an evaluation environment.
        dataloader (DataLoader): the dataloader to use for evaluation.
        evaluation_suite (Optional[EvaluationSuite], optional): The evaluation suite. Defaults to None.

    Returns:
        tuple[xr.Dataset, dict[str, Any]]: the output dataset and the evaluation results.
    """
    if evaluation_suite is None:
        evaluation_suite = {"eval_fn_inputs": lambda eval_fn_input: eval_fn_input}

    def eval_env_return_xarray(env: ModuleEvalEnv, dataset: XarrayPreppedDataset) -> xr.Dataset:
        ds = dataset.ds
        inputs, _ = ds.popsim_ml.prep_inputs_and_targets()
        inputs_spec = jax.tree.map(lambda _: 0, inputs)
        sol = jax.vmap(env, in_axes=(inputs_spec,))(inputs)
        ds_out = solution_to_xarray(sol, multi_simulation=True)

        # Rename the dimensions to match the input dataset.
        ds_out = ds_out.rename({DEFAULT_SIM_DIM_NAME: DEFAULT_SAMPLE_DIM})
        ds_out = ds_out.assign_coords({DEFAULT_SAMPLE_DIM: ds.popsim_ml.sample_coord})
        ds_out = ds_out.rename_dims({DEFAULT_TIME_DIM_NAME: ds.popsim_ml.training_metadata.time_dep_metadata.time_dim})
        return ds_out

    sim_outs_and_batches = [(eval_env_return_xarray(env, batch), batch) for batch in dataloader]
    sim_outs = [sim_out for sim_out, _ in sim_outs_and_batches]

    # The evaluation process can get the order of the samples wrong, so we need to reindex the output dataset.
    ds_sim = xr.concat(sim_outs, dim=DEFAULT_SAMPLE_DIM)
    ds_sim = ds_sim.reindex_like(dataloader.dataloader.dataset.ds)

    eval_fn_input = EvalFnInput(model=env, dataloader=dataloader, output_ds=ds_sim)

    out = {key: eval_fn(eval_fn_input) for key, eval_fn in evaluation_suite.items()}
    return out


@eqx.filter_jit
def model_eval_and_loss(
    model: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
) -> float:
    """Run the model on the inputs and compute the loss. Supports POPSIM simulation modules and also time-independent modules.

    Args:
        model (TrainableModel): the model to evaluate.
        loss_fn (LossFunction): the loss function to use.
        inputs (PyTree[Array]): the inputs to the model.
        targets (PyTree[Array]): the targets to compare the model output to in the loss function.

    Returns:
        float: the loss.
    """
    if isinstance(model, ModuleEvalEnv):
        loss_fn = eqx.error_if(
            loss_fn, not isinstance(loss_fn, IntegralLoss), "When using a ModuleEvalEnv, the loss function must be an IntegralLoss."
        )
        # When using a ModuleEvalEnv, the loss function is an IntegralLoss, which requires special handling.
        output = model(inputs)  # Output is a diffrax solution.
        loss = loss_fn(output.ys["output"], targets, inputs.time)
    else:
        output = model(inputs)
        loss = loss_fn(output, targets)
    return loss


def batched_model_eval_and_loss(
    model: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
) -> Array:
    """A thin vectorized wrapper around model_eval_and_loss that computes the loss for each sample in a batch.

    Args:
        model (TrainableModel): the model to evaluate.
        loss_fn (LossFunction): the loss function to use.
        inputs (PyTree[Array]): the inputs to the model.
        targets (PyTree[Array]): the targets to compare the model output to in the loss function.

    Returns:
        Array: the vector of losses for the samples.
    """
    inputs_spec, targets_spec = jax.tree.map(lambda _: 0, (inputs, targets))
    losses = jax.vmap(model_eval_and_loss, in_axes=(None, None, inputs_spec, targets_spec))(model, loss_fn, inputs, targets)
    return losses


def batch_loss(
    trainable: TrainableModel,
    static: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
) -> float:
    """Computes the mean batch loss with support for partitioning the model into trainable and static parts.

    Args:
        trainable (TrainableModel): the trainable part of the model.
        static (TrainableModel): the static part of the model.
        loss_fn (LossFunction): the loss function to use.
        inputs (PyTree[Array]): the inputs to the model.
        targets (PyTree[Array]): the targets to compare the model output to in the loss function.

    Returns:
        float: the mean batch loss.
    """
    model = eqx.combine(trainable, static)
    losses = batched_model_eval_and_loss(model, loss_fn, inputs, targets)
    return losses.mean()


def make_val_loss_eval_fn(
    loss_fn: LossFunction,
) -> EvaluationFn:
    """Given a loss function, generate an evaluation function that computes the loss on a dataset.

    Args:
        loss_fn (LossFunction): the loss function to use.

    Returns:
        EvaluationFn: the evaluation function.
    """

    def eval_fn(inp: EvalFnInput) -> float:
        loss_vecs = []
        for batch in inp.dataloader:
            inputs, targets = batch.ds.popsim_ml.prep_inputs_and_targets()
            loss_vec = batched_model_eval_and_loss(
                inp.model,
                loss_fn,
                inputs,
                targets,
            )
            loss_vecs.append(loss_vec)
        loss_vec = jnp.concatenate(loss_vecs)
        out = {
            "mean": loss_vec.mean(),
            "vec": loss_vec,
        }

        return out

    return eval_fn
