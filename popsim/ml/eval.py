from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jaxtyping import Array, PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import DEFAULT_SAMPLE_DIM, DataLoader, XarrayPreppedDataset
from popsim.ml.envs import ModuleEvalEnv, ModuleTrainingEnv
from popsim.ml.loss import IntegralLoss, LossFunction
from popsim.xarray_utils import pytree_to_xarray, run_function_with_dim_removed

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
        # The evaluation pipeline compares model outputs (in-memory JAX) to this,
        # so we need to make sure it's loaded into memory as well
        return self.dataloader.ds.load()


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

    # Best practice for making sure things like dropout are disabled.
    # https://docs.kidger.site/equinox/api/nn/inference/
    model = eqx.nn.inference_mode(model)

    def eval_env_return_xarray(env: ModuleEvalEnv, dataset: XarrayPreppedDataset) -> xr.Dataset:
        ds_in = dataset.ds
        inputs, _ = dataset.get_inputs_and_targets()
        inputs_spec = jax.tree.map(lambda _: 0, inputs)
        vec_env = jax.vmap(env, in_axes=(inputs_spec,))

        sol = run_function_with_dim_removed(vec_env, (inputs,), DEFAULT_SAMPLE_DIM)

        training_meta = dataset.training_metadata

        ds_out = pytree_to_xarray(sol.ys, [training_meta.sample_dim, training_meta.time_dep_metadata.time_dim], {})

        # Make sure the output data has access to the same coordinates as the input data.
        ds_out = ds_out.assign_coords(ds_in.coords)

        return ds_out

    def eval_model_return_xarray(model: TrainableModel, dataset: XarrayPreppedDataset) -> xr.Dataset:
        inputs, _ = dataset.get_inputs_and_targets()
        inputs_spec = jax.tree.map(lambda _: 0, inputs)
        fn = jax.vmap(model, in_axes=(inputs_spec,))

        out = run_function_with_dim_removed(fn, (inputs,), DEFAULT_SAMPLE_DIM)

        ds_out = pytree_to_xarray(out, base_dims=[DEFAULT_SAMPLE_DIM], base_coords={DEFAULT_SAMPLE_DIM: dataset.sample_coord})

        # Re-assign the sample coordinates to the output dataset.
        # Drop existing multi-index level coordinates first to avoid inconsistent state.
        sample_dim = dataset.training_metadata.sample_dim
        coords_to_drop = [c for c in ds_out.coords if c in dataset.sample_coord.coords and c != sample_dim]
        if coords_to_drop:
            ds_out = ds_out.drop_vars(coords_to_drop)
        ds_out = ds_out.assign_coords({sample_dim: dataset.sample_coord})
        return ds_out

    if isinstance(model, ModuleEvalEnv | ModuleTrainingEnv):
        eval_fn = eval_env_return_xarray
    else:
        eval_fn = eval_model_return_xarray

    sim_outs_and_batches = [(eval_fn(model, batch), batch) for batch in dataloader]

    sim_outs = [sim_out for sim_out, _ in sim_outs_and_batches]

    ds_sim = xr.concat(sim_outs, dim=DEFAULT_SAMPLE_DIM)

    ds_sim = ds_sim.reindex_like(dataloader.ds)

    eval_fn_input = EvalData(model=model, dataloader=dataloader, output_ds=ds_sim)
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
    inputs_spec, targets_spec = jax.tree.map(lambda _: 0, (inputs, targets))

    vec_model_eval_and_loss = jax.vmap(model_eval_and_loss, in_axes=(None, None, inputs_spec, targets_spec))

    losses = run_function_with_dim_removed(vec_model_eval_and_loss, (model, loss_fn, inputs, targets), DEFAULT_SAMPLE_DIM)

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


def masked_batch_loss(
    trainable: TrainableModel,
    static: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
    sample_mask: Array,
) -> float:
    """Mean batch loss over only the samples where sample_mask is True.

    Masking the loss alone is not enough to mask a sample's gradient: if the forward pass
    at a bad sample is non-finite, its cotangent is NaN and 0 * NaN = NaN poisons the whole
    batch gradient. So the masked-out samples' inputs and targets are first replaced with
    those of the first masked-in sample, keeping the differentiated computation finite
    everywhere, and then given zero weight in the mean. Masked-out samples therefore
    contribute exactly zero loss and zero gradient.

    Args:
        trainable (TrainableModel): the trainable part of the model.
        static (TrainableModel): the static part of the model.
        loss_fn (LossFunction): the loss function to use.
        inputs (PyTree[Array]): the inputs to the model, leading dim is the sample dim.
        targets (PyTree[Array]): the targets, leading dim is the sample dim.
        sample_mask (Array): boolean vector over the sample dim, True = include the sample.
            Must have at least one True entry (the caller is expected to check host-side).

    Returns:
        float: the mean batch loss over the masked-in samples.
    """
    model = eqx.combine(trainable, static)
    good_idx = jnp.argmax(sample_mask)

    def _replace_masked_out(leaf):
        mask = sample_mask.reshape((leaf.shape[0],) + (1,) * (leaf.ndim - 1))
        return jnp.where(mask, leaf, jax.lax.dynamic_index_in_dim(leaf, good_idx, keepdims=True))

    safe_inputs, safe_targets = jax.tree.map(_replace_masked_out, (inputs, targets))
    losses = batched_model_eval_and_loss(model, loss_fn, safe_inputs, safe_targets)
    weights = sample_mask.astype(losses.dtype)
    return jnp.sum(losses * weights) / jnp.sum(weights)


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
