import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import DEFAULT_SAMPLE_DIM, DataLoader, XarrayPreppedDataset
from popsim.ml.envs import ModuleEvalEnv, ModuleTrainingEnv
from popsim.ml.loss import IntegralLoss, LossFunction
from popsim.xarray_utils import DEFAULT_TIME_DIM_NAME, pytree_to_xarray, run_function_with_dim_removed

if TYPE_CHECKING:
    import diffrax

"""
This module contains utilities for evaluating models on data.
"""


class EvalData:
    """A model, the dataloader it is evaluated on, and the model outputs.

    output_ds is built on first access,
    so evaluation functions that only need the model and dataloader skip the full forward pass.
    """

    def __init__(self, model: TrainableModel, dataloader: DataLoader, output_ds_builder: Callable[[], xr.Dataset]):
        self.model = model
        self.dataloader = dataloader
        self._output_ds_builder = output_ds_builder

    @functools.cached_property
    def output_ds(self) -> xr.Dataset:
        """The output of the model converted to an xarray dataset."""
        return self._output_ds_builder()

    @property
    def input_ds(self):
        """
        Get the xarray dataset that was used as input to the module evaluation.
        """
        # The evaluation pipeline compares model outputs (in-memory JAX) to this,
        # so we need to make sure it's loaded into memory as well
        return self.dataloader.ds.load()


# An evaluation function is a function that takes an EvalData and returns a value.
EvaluationFn = Callable[[EvalData], Any]

# An evaluation suite is defined as a dictionary of evaluation functions
EvaluationSuite = dict[str, EvaluationFn]


def eval_model_on_data(model: TrainableModel, dataloader: DataLoader) -> EvalData:
    """Evaluate a module on data from a dataloader.

    The forward pass runs when output_ds is first accessed on the returned EvalData.

    Args:
        model (TrainableModel): the model, or a module wrapped in an evaluation environment.
        dataloader (DataLoader): the dataloader to use for evaluation.

    Returns:
        EvalData: evaluation results.
    """

    # Best practice for making sure things like dropout are disabled.
    # https://docs.kidger.site/equinox/api/nn/inference/
    model = eqx.nn.inference_mode(model)

    def eval_env_return_xarray(env: ModuleEvalEnv, dataset: XarrayPreppedDataset) -> xr.Dataset:
        ds_in = dataset.ds
        inputs, _ = dataset.get_inputs_and_targets()

        sol = run_function_with_dim_removed(env, (inputs,), DEFAULT_SAMPLE_DIM, in_axes=(0,))

        training_meta = dataset.training_metadata
        time_dim = training_meta.time_dep_metadata.time_dim

        ds_out = pytree_to_xarray(sol.ys, [training_meta.sample_dim, time_dim], {})

        # xr.Variable outputs get their time dimension named by the simulation machinery.
        # Rename it to match the dataset's time dimension used for the array outputs,
        # unless they already match in case we don't need to do anything
        if DEFAULT_TIME_DIM_NAME in ds_out.dims and time_dim != DEFAULT_TIME_DIM_NAME:
            ds_out = ds_out.rename({DEFAULT_TIME_DIM_NAME: time_dim})

        # Make sure the output data has access to the same coordinates as the input data.
        ds_out = ds_out.assign_coords(ds_in.coords)

        return ds_out

    def eval_model_return_xarray(model: TrainableModel, dataset: XarrayPreppedDataset) -> xr.Dataset:
        inputs, _ = dataset.get_inputs_and_targets()

        out = run_function_with_dim_removed(model, (inputs,), DEFAULT_SAMPLE_DIM, in_axes=(0,))

        ds_out = pytree_to_xarray(out, base_dims=[DEFAULT_SAMPLE_DIM], base_coords={DEFAULT_SAMPLE_DIM: dataset.sample_coord})

        # Re-assign the sample coordinates to the output dataset.
        # Drop existing multi-index level coordinates first to avoid inconsistent state.
        sample_dim = dataset.training_metadata.sample_dim
        coords_to_drop = [c for c in ds_out.coords if c in dataset.sample_coord.coords and c != sample_dim]
        if coords_to_drop:
            # Also drop the multi-index dim coordinate itself
            # (deleting only its levels is deprecated in xarray)
            # It is restored by the assign_coords below
            ds_out = ds_out.drop_vars([*coords_to_drop, sample_dim])
        ds_out = ds_out.assign_coords({sample_dim: dataset.sample_coord})
        return ds_out

    if isinstance(model, ModuleEvalEnv | ModuleTrainingEnv):
        eval_fn = eval_env_return_xarray
    else:
        eval_fn = eval_model_return_xarray

    def _build_output_ds() -> xr.Dataset:
        sim_outs = [eval_fn(model, batch) for batch in dataloader]

        ds_sim = xr.concat(sim_outs, dim=DEFAULT_SAMPLE_DIM)

        # Drop padded duplicate samples from a pad_last dataloader before reindexing
        ds_sim = ds_sim.isel({DEFAULT_SAMPLE_DIM: slice(0, dataloader.dataset.n_samples)})

        ds_sim = ds_sim.reindex_like(dataloader.ds)
        return ds_sim

    eval_fn_input = EvalData(model=model, dataloader=dataloader, output_ds_builder=_build_output_ds)
    return eval_fn_input


def run_evals(model: TrainableModel, dataloader: DataLoader, evaluation_suite: EvaluationSuite | None = None) -> dict[str, Any] | EvalData:
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
    losses = run_function_with_dim_removed(
        model_eval_and_loss, (model, loss_fn, inputs, targets), DEFAULT_SAMPLE_DIM, in_axes=(None, None, 0, 0)
    )

    return losses


def masked_batch_loss(
    trainable: TrainableModel,
    static: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
    sample_mask: Array,
) -> tuple[Array, Array]:
    """Mean batch loss over the samples where sample_mask is True, plus every per-sample loss.

    Masked-out samples add no loss and no gradient only while their own loss is finite,
    because 0 * NaN = NaN in both the forward and backward pass.
    Callers must overwrite non-finite samples with finite ones before masking them out.

    Args:
        trainable (TrainableModel): the trainable part of the model.
        static (TrainableModel): the static part of the model.
        loss_fn (LossFunction): the loss function to use.
        inputs (PyTree[Array]): the inputs to the model.
        targets (PyTree[Array]): the targets to compare the model output to in the loss function.
        sample_mask (Array): boolean vector over the sample dim, True = include the sample.
            Must have at least one True entry.

    Returns:
        tuple[Array, Array]: the masked mean loss and the vector of per-sample losses.
    """
    model = eqx.combine(trainable, static)
    sample_losses = batched_model_eval_and_loss(model, loss_fn, inputs, targets)
    sample_weights = sample_mask.astype(sample_losses.dtype)
    masked_mean_loss = jnp.sum(sample_losses * sample_weights) / jnp.sum(sample_weights)
    return masked_mean_loss, sample_losses


def make_val_loss_eval_fn(
    loss_fn: LossFunction,
) -> EvaluationFn:
    """Given a loss function, generate an evaluation function that computes the loss on a dataset.

    Args:
        loss_fn (LossFunction): the loss function to use.

    Returns:
        EvaluationFn: the evaluation function.
    """

    # A fresh closure per suite gets its own filter_jit cache, reused by every validation of this run.
    # A cache shared across unrelated models can raise instead of retracing,
    # since some statics (e.g. xarray attrs holding numpy arrays) raise on __eq__
    def _eval_and_loss(model, loss_fn, inputs, targets):
        return batched_model_eval_and_loss(model, loss_fn, inputs, targets)

    jit_eval_and_loss = eqx.filter_jit(_eval_and_loss)

    def eval_fn(inp: EvalData) -> float:
        loss_vecs = []
        for batch in inp.dataloader:
            inputs, targets = batch.get_inputs_and_targets()
            loss_vec = jit_eval_and_loss(
                inp.model,
                loss_fn,
                inputs,
                targets,
            )
            loss_vecs.append(loss_vec)
        loss_vec = jnp.concatenate(loss_vecs)
        # Drop padded duplicate samples from a pad_last dataloader
        loss_vec = loss_vec[: inp.dataloader.dataset.n_samples]
        out = {
            "mean": loss_vec.mean(),
            "vec": loss_vec,
        }

        return out

    return eval_fn
