import equinox as eqx
import jax
import xarray as xr
from jax_dataloader import DataLoader

from popsim import interp
from popsim.ml.dataloading import XarrayPreppedDataset
from popsim.ml.envs import ModuleEvalEnv
from popsim.ml.types import EvalInput
from popsim.ml.utils import _repeat_time_hack
from popsim.simulate import SimInput, _diffrax_simulate, _vec_simulate
from popsim.xarray_utils import solution_to_xarray


@eqx.filter_jit
def prep_batch(env: ModuleEvalEnv, eval_input: EvalInput):
    params_struct = env.create_params(data=eval_input.sim_input.params)
    params_struct_spec = jax.tree.map(lambda _: 0, params_struct)

    def interp_params_one_episode(t, p):
        return interp.interp(_repeat_time_hack(t), p, interp.InterpType.RECTILINEAR)

    params_interped = jax.vmap(interp_params_one_episode, in_axes=(0, params_struct_spec))(eval_input.sim_input.time, params_struct)

    sim_input = SimInput(
        time=eval_input.sim_input.time,
        initial_state=env.create_state(data=eval_input.sim_input.initial_state),
        params=params_interped,
    )
    return sim_input


def simulate_batch(env: ModuleEvalEnv, dataset: XarrayPreppedDataset) -> xr.Dataset:
    eval_input = dataset.to_eval_input()
    sim_input = prep_batch(env, eval_input)
    sol = _vec_simulate(env.module, sim_input, _diffrax_simulate)

    ds = solution_to_xarray(sol, multi_simulation=True)
    ds = ds.rename({"simulation": "sample"})
    ds = ds.assign_coords(sample=dataset.sample_coords)
    return ds


def eval_module_on_dataset(env: ModuleEvalEnv, dataloader: DataLoader, evaluation_suite=None) -> xr.Dataset:
    if evaluation_suite is None:
        evaluation_suite = {}
    sim_outs_and_batches = [(simulate_batch(env, batch), batch) for batch in dataloader]
    batches_in = [batch.ds for _, batch in sim_outs_and_batches]
    sim_outs = [sim_out for sim_out, _ in sim_outs_and_batches]
    ds_data = xr.concat(batches_in, dim="sample")
    ds_sim = xr.concat(sim_outs, dim="sample")
    out = {
        "ds_data": ds_data,
        "ds_sim": ds_sim,
    }
    for key, eval_fn in evaluation_suite.items():
        out[key] = eval_fn(ds_data, ds_sim)
    return out
