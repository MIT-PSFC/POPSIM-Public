import typing
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import xarray as xr
from jax_dataloader import DataLoader
from jaxtyping import Array, PyTree

from popsim.ml._types import TrainableModel
from popsim.ml.dataloading import XarrayPreppedDataset
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.loss import IntegralLoss, LossFunction
from popsim.xarray_utils import solution_to_xarray


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
EvaluationFn = typing.Callable[[EvalFnInput], typing.Any]

# An evaluation suite is defined as a dictionary of evaluation functions
EvaluationSuite = dict[str, typing.Callable[[xr.Dataset, xr.Dataset], typing.Any]]


def eval_module_on_dataset(
    env: ModuleEvalEnv, dataloader: DataLoader, evaluation_suite: typing.Optional[EvaluationSuite] = None
) -> tuple[xr.Dataset, dict[str, typing.Any]]:
    if evaluation_suite is None:
        evaluation_suite = {}

    def eval_env_return_xarray(env: ModuleEvalEnv, dataset: XarrayPreppedDataset) -> xr.Dataset:
        inputs, _ = dataset.prep_inputs_and_targets()
        inputs_spec = jax.tree.map(lambda _: 0, inputs)
        sol = jax.vmap(env, in_axes=(inputs_spec,))(inputs)
        ds_out = solution_to_xarray(sol, multi_simulation=True)
        ds_out = ds_out.rename({"simulation": "sample"})
        ds_out = ds_out.assign_coords(sample=dataset.sample_coords)
        return ds_out

    sim_outs_and_batches = [(eval_env_return_xarray(env, batch), batch) for batch in dataloader]
    sim_outs = [sim_out for sim_out, _ in sim_outs_and_batches]

    # The evaluation process can get the order of the samples wrong, so we need to reindex the output dataset.
    ds_sim = xr.concat(sim_outs, dim="sample")
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


@eqx.filter_jit
def batched_model_eval_and_loss(
    model: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
) -> Array:
    inputs_spec, targets_spec = jax.tree.map(lambda _: 0, (inputs, targets))
    losses = jax.vmap(model_eval_and_loss, in_axes=(None, None, inputs_spec, targets_spec))(model, loss_fn, inputs, targets)
    eqx.error_if(losses, jnp.any(jnp.isnan(losses)), "NaN values found in loss values.")
    return losses


@eqx.filter_value_and_grad
def batch_loss_and_grad(
    trainable: TrainableModel,
    static: TrainableModel,
    loss_fn: LossFunction,
    inputs: PyTree[Array],
    targets: PyTree[Array],
) -> float:
    model = eqx.combine(trainable, static)
    losses = batched_model_eval_and_loss(model, loss_fn, inputs, targets)
    return losses.mean()


def make_val_loss_eval_fn(
    loss_fn: LossFunction,
):
    def eval_fn(inp: EvalFnInput) -> float:
        loss_vecs = []
        for batch in inp.dataloader:
            inputs, targets = batch.prep_inputs_and_targets()
            loss_vec = batched_model_eval_and_loss(
                inp.model,
                loss_fn,
                inputs,
                targets,
            )
            loss_vecs.append(loss_vec)
        return jnp.concatenate(loss_vecs).mean()

    return eval_fn
