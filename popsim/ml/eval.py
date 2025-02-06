from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import DEFAULT_SAMPLE_DIM, DataLoader, XarrayPreppedDataset
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.loss import IntegralLoss, LossFunction
from popsim.xarray_utils import (
    DEFAULT_SIM_DIM_NAME,
    DEFAULT_TIME_DIM_NAME,
    pytree_to_xarray,
    run_function_with_dim_removed,
    solution_to_xarray,
)

if TYPE_CHECKING:
    import diffrax

"""
This module contains utilities for evaluating models on data.
"""


class EvalData(NamedTuple):
    model: TrainableModel  # The model to train.
    dataloader: DataLoader  # DataLoader that was used to evaluate the module.
    output_ds: xr.Dataset  # The output of the model converted to an xarray dataset.

    @property
    def input_ds(self):
        """
        Get the xarray dataset that was used as input to the module evaluation.
        """
        return self.dataloader.ds


# An evaluation function is a function that takes an EvalData and returns a value.
EvaluationFn = Callable[[EvalData], Any]

# An evaluation suite is defined as a dictionary of evaluation functions
EvaluationSuite = dict[str, EvaluationFn]


def eval_model_on_data(model: TrainableModel, dataloader: DataLoader) -> EvalData:
    """Evaluate a module on data from a dataloader.

    Args:
        env (TrainableModel): the module wrapped in an evaluation environment.
        dataloader (DataLoader): the dataloader to use for evaluation.

    Returns:
        EvalData: evaluation results.
    """

    def eval_env_return_xarray(env: ModuleEvalEnv, dataset: XarrayPreppedDataset) -> xr.Dataset:
        inputs, _ = dataset.get_inputs_and_targets()
        inputs_spec = jax.tree.map(lambda _: 0, inputs)
        vec_env = jax.vmap(env, in_axes=(inputs_spec,))

        sol = run_function_with_dim_removed(vec_env, (inputs,), DEFAULT_SAMPLE_DIM)

        ds_out = solution_to_xarray(sol, multi_simulation=True)

        # Rename the dimensions to match the input dataset.
        ds_out = ds_out.rename({DEFAULT_SIM_DIM_NAME: DEFAULT_SAMPLE_DIM})
        ds_out = ds_out.assign_coords({DEFAULT_SAMPLE_DIM: dataset.sample_coord})
        ds_out = ds_out.rename_dims({DEFAULT_TIME_DIM_NAME: dataset.training_metadata.time_dep_metadata.time_dim})
        return ds_out

    def eval_model_return_xarray(model: TrainableModel, dataset: XarrayPreppedDataset) -> xr.Dataset:
        inputs, _ = dataset.get_inputs_and_targets()
        inputs_spec = jax.tree.map(lambda _: 0, inputs)
        fn = jax.vmap(model, in_axes=(inputs_spec,))

        out = run_function_with_dim_removed(fn, (inputs,), DEFAULT_SAMPLE_DIM)

        ds_out = pytree_to_xarray(out, base_dims=[DEFAULT_SAMPLE_DIM], base_coords={DEFAULT_SAMPLE_DIM: dataset.sample_coord})
        return ds_out

    if isinstance(model, ModuleEvalEnv):
        eval_fn = eval_env_return_xarray
    else:
        eval_fn = eval_model_return_xarray

    sim_outs_and_batches = [(eval_fn(model, batch), batch) for batch in dataloader]
    sim_outs = [sim_out for sim_out, _ in sim_outs_and_batches]

    # The evaluation process can get the order of the samples wrong, so we need to reindex the output dataset.
    ds_sim = xr.concat(sim_outs, dim=DEFAULT_SAMPLE_DIM)
    ds_sim = ds_sim.reindex_like(dataloader.ds)

    eval_fn_input = EvalData(model=model, dataloader=dataloader, output_ds=ds_sim)
    return eval_fn_input


def run_evals(
    model: TrainableModel, dataloader: DataLoader, evaluation_suite: Optional[EvaluationSuite] = None
) -> Union[dict[str, Any], EvalData]:
    """Given a model, a dataloader, and an evaluation suite, run the evaluation suite on the model and return the results.

    Args:
        model (TrainableModel): the model to evaluate.
        dataloader (DataLoader): the dataloader to use for evaluation.
        evaluation_suite (Optional[EvaluationSuite], optional): Optional evaluation suite to run. This is a dictionary of evaluation functions that take in an EvalData structure and returns the evaluation results. If None is provided, just return the EvalData generated. Defaults to None.

    Returns:
        Union[dict[str, Any], EvalData]: the evaluation results.
    """
    eval_fn_input = eval_model_on_data(model, dataloader)
    if evaluation_suite is None:
        return eval_fn_input

    eval_results = {key: eval_fn(eval_fn_input) for key, eval_fn in evaluation_suite.items()}
    return eval_results


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
        output: diffrax.Solution = model(inputs)
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

    def eval_fn(inp: EvalData) -> float:
        loss_vecs = []
        for batch in inp.dataloader:
            inputs, targets = batch.get_inputs_and_targets()
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
