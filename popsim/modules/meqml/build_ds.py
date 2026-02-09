import glob

import fire
import xarray as xr

from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.interfaces.meq2xarray.meq2xarray import tcv_db_to_xr, tcv_fbt_to_xr
from popsim.math_utils import signed_log
from popsim.modules.meqml.fast_obs.config import FAST_OBS_CONFIG
from popsim.modules.meqml.fbt.config import FBT_SURROGATE_CONFIG
from popsim.modules.meqml.fge.config import FGE_CONFIG
from popsim.modules.meqml.fge_mat.config import FGE_MAT_CONFIG
from popsim.modules.meqml.fgs.config import FGS_CONFIG
from popsim.modules.meqml.liuqe.config import LIUQE_CONFIG
from popsim.modules.meqml.utils import compute_eigs


def build_fbt_ds(glob_str, zarr_path, extend_existing):
    paths = glob.glob(glob_str)
    config = FBT_SURROGATE_CONFIG

    def process_fn(f):
        ds = tcv_fbt_to_xr(f)

        all_vars = (
            config["dataloader_config"]["input_vars"]
            + config["dataloader_config"]["target_vars"]
            + config["dataloader_config"]["extra_vars"]
        )

        all_vars = [v for v in all_vars if v in ds.data_vars]
        ds = ds[all_vars]
        return ds

    build_tensorized_dataset(
        process_fn=process_fn,
        identifiers=paths,
        zarr_path=zarr_path,
        time_dim="time_idx",
        episode_dim="shot",
        extend_existing=extend_existing,
        mb_per_chunk=50,
    )


def build_liuqe_ds(glob_str, zarr_path, extend_existing: bool = False):
    paths = glob.glob(glob_str)

    def process_fn(f):
        dt = tcv_db_to_xr(f)
        ds = dt["liuqe"].to_dataset().isel(pq=-1).isel(pQ=-1)
        all_vars = set(
            LIUQE_CONFIG["dataloader_config"]["input_vars"]
            + LIUQE_CONFIG["dataloader_config"]["target_vars"]
            + LIUQE_CONFIG["dataloader_config"]["extra_vars"]
            + FGS_CONFIG["dataloader_config"]["input_vars"]
            + FGS_CONFIG["dataloader_config"]["target_vars"]
            + FGS_CONFIG["dataloader_config"]["extra_vars"]
            + FAST_OBS_CONFIG["dataloader_config"]["input_vars"]
            + FAST_OBS_CONFIG["dataloader_config"]["target_vars"]
            + FAST_OBS_CONFIG["dataloader_config"]["extra_vars"]
        )
        ds = ds[all_vars]
        return ds

    build_tensorized_dataset(
        process_fn=process_fn,
        identifiers=paths,
        zarr_path=zarr_path,
        time_dim="time_idx",
        episode_dim="shot",
        extend_existing=extend_existing,
        mb_per_chunk=50,
    )


def build_fge_ds(glob_str, zarr_path, extend_existing: bool = False):
    paths = glob.glob(glob_str)
    all_vars = set(
        FGE_CONFIG["dataloader_config"]["input_vars"]
        + FGE_CONFIG["dataloader_config"]["target_vars"]
        + FGE_CONFIG["dataloader_config"]["extra_vars"]
        + FGE_MAT_CONFIG["dataloader_config"]["input_vars"]
        + FGE_MAT_CONFIG["dataloader_config"]["target_vars"]
        + FGE_MAT_CONFIG["dataloader_config"]["extra_vars"]
    )

    def strip_ags(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.reindex(state_out=ds.state.values)
        non_ag_states = [v for v in ds.state.values if "ag" not in v]
        ds = ds.sel(state=non_ag_states).sel(state_out=non_ag_states)
        return ds

    def rename_and_drop_extra_inputs(ds: xr.Dataset):
        inputs_drop = ["Ini_001_S", "dCodt_bp_001", "dCodt_qA_001"]

        ds = ds.sel(input=[v for v in ds.input.values if v not in inputs_drop])

        renames_manual = {
            "Co_bp_001": "betap",
            "Co_qA_001": "qA",
        }
        renames_va = {v: v.replace("Va_", "") for v in ds.input.values if v.startswith("Va_")}

        renames = {**renames_manual, **renames_va}

        ds.coords["input"] = [renames[v] if v in renames else v for v in ds.input.values]
        return ds

    def process_fn(f):
        dt = tcv_db_to_xr(f)
        liuqeds = dt["liuqe"].to_dataset()
        liuqe_vars = [v for v in liuqeds if v in all_vars]
        liuqeds = liuqeds[liuqe_vars]

        liuqeds = liuqeds.drop_duplicates("time")

        ssds = dt["ss"].to_dataset()

        # Remove "ag" states.
        ssds = strip_ags(ssds)

        # Rename and drop extra input variables (e.g. Ini_001_S, dCodt_bp_001, dCodt_qA_001)
        ssds = rename_and_drop_extra_inputs(ssds)

        ssds = compute_eigs(ssds)

        # Drop all samples with second eigenvalue >= 0.0.
        ssds = ssds.where(ssds["Aeigvals"].sel(eigenmode=slice(1, None)).max("eigenmode") < 0.0, drop=True)

        ssds["Aeigvals_slog"] = xr.apply_ufunc(signed_log, ssds["Aeigvals"])

        # Loading the full C matrix gets a bit crazy. Only keeping Ia and Iu.
        ia_outs = [v for v in ssds.output.values if v.startswith("Ia")]
        iu_outs = [v for v in ssds.output.values if v.startswith("Iu")]
        if len(iu_outs) > 38:
            raise ValueError("Too many Iu outputs.")
        ssds = ssds.sel(output=ia_outs + iu_outs)

        ssds_vars = [v for v in ssds.data_vars if v in all_vars]

        ssds = ssds[ssds_vars]

        liuqeds = liuqeds.sel(time=ssds.time.values, method="nearest")
        liuqeds = liuqeds.interp_like(ssds)

        acts = dt["acts"].ds["Va"].sel(time=dt["ss"].ds.time, method="nearest")

        acts = acts.interp_like(ssds)

        acts = acts.fillna(0.0)

        ds = xr.merge([liuqeds, ssds, acts])

        max_eigvals = ds["Aeigvals"].max("eigenmode")

        ds = ds.where((max_eigvals < FGE_MAT_CONFIG["dataloader_config"]["max_vgr_threshold"]) & (0.0 < max_eigvals), drop=True)

        # Re-scale B to take in normalized voltages.
        for coil in ds.active_coils.data:
            ds["B"].loc[dict(input=coil)] = ds["B"].sel(input=coil).data * dt["liuqe"].ds["L.G.Vamax"].sel(active_coils=coil).data

        return ds

    build_tensorized_dataset(
        process_fn=process_fn,
        identifiers=paths,
        zarr_path=zarr_path,
        time_dim="time_idx",
        episode_dim="shot",
        extend_existing=extend_existing,
        mb_per_chunk=50,
    )


if __name__ == "__main__":
    fire.Fire()
