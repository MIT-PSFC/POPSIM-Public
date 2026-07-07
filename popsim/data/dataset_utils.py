import gc
import os
import shutil
from collections.abc import Callable, Iterable
from typing import Any

import jax
import loguru
import numpy as np
import xarray as xr
import zarr
from tqdm import tqdm

GC_INTERVAL = 40  # Force garbage collection every 40 shots


def build_tensorized_dataset(  # noqa: PLR0912
    process_fn: Callable[[Any], xr.Dataset],
    identifiers: Iterable[Any],
    zarr_path: os.PathLike,
    time_dim: str,
    episode_dim: str,
    extend_existing: bool = False,
    episodes_per_chunk: int | None = None,
    mb_per_chunk: int | None = 10,
    dim_sizes: dict[str, int] | None = None,
) -> xr.Dataset:
    """Build a tensorized multi-episode dataset that can then be used for training or evaluation.
    The user provides a function that processes data for a single episode, which returns an xarray Dataset for a single episode. This function will then build up the multi-episode dataset. This function was built with the intention of only needing to load a single episode at a time, allowing us to build up a dataset that is too large to fit in memory.

    Args:
        process_fn (Callable[[Any], xr.Dataset]): A function that takes in a episode identifier (e.g. a file path, a pulse number, or even a xarray Dataset) and returns an xarray Dataset for a single episode.
        identifiers (Iterable[Any]): An iterable that will be looped over, with each item being passed to process_fn to build up the dataset.
        zarr_path (os.PathLike): path to the zarr store where the dataset will be saved.
        time_dim (str): The name of the time dimension in the dataset.
        episode_dim (str): The name of the episode dimension in the dataset.
        extend_existing (bool, optional): If zarr_path already exists and this is true, we will try to extend the existing zarr_path. Defaults to False.
        episodes_per_chunk (Optional[int], optional): The number of episodes per chunk in storage. If this is not None, then this function will rechunk the built Zarr store once all the files are added. Defaults to None.
        mb_per_chunk (Optional[int], optional): If specified, the resulting Zarr stored will be chunked along the episodes dimension with size max(1, int(mb_per_chunk / mean_mb_per_episode)). Defaults to 10.
        dim_sizes (Optional[dict[str, int]], optional): Upper bound on the size of each non-episode dimension across all episodes. If provided, every episode is padded with NaNs to these sizes before being written, so the store never needs to be extended when a later episode is larger than earlier ones. Any overshoot from the upper bound is trimmed during the final rechunking pass (trailing slices that are NaN across all episodes and variables are dropped). An error is raised if an episode exceeds one of these sizes. Defaults to None.

    Raises:
        ValueError: If zarr_path already exists and extend_existing is False, this function will raise an error.

    Returns:
        xr.Dataset: The xarray Dataset that was built up.
    """

    if not zarr_path.endswith(".zarr"):
        raise ValueError(f"Provided zarr_path must end with .zarr, but got {zarr_path}")

    if os.path.exists(zarr_path) and extend_existing is False:
        raise ValueError(
            f"Zarr store at {zarr_path} already exists and extend_existing is False. Please remove it or set extend_existing to True."
        )
    if mb_per_chunk is not None and episodes_per_chunk is not None:
        raise ValueError("Please specify either mb_per_chunk or episodes_per_chunk, not both.")

    def get_and_preprocess(identifier: Any) -> xr.Dataset | None:
        try:
            return process_fn(identifier)
        except Exception as e:
            # Warn the user.
            loguru.logger.warning(f"Error processing {identifier}: {e}")
            return None

    atleast_one_success = False  # Keep track of whether at least one dataset was successfully processed.
    store_time_dim_size = None  # Keep track of the size of the time dimension in the zarr store.

    # Iterate over the identifiers and process them one by one to build the dataset.
    for i, it in enumerate(tqdm(identifiers, desc="Building the dataset...")):
        ds = get_and_preprocess(it)
        if ds is not None:
            success = add_to_zarr_store(
                ds, zarr_path, time_dim, episode_dim, store_time_dim_size=store_time_dim_size, dim_sizes=dim_sizes
            )
            if success and not store_time_dim_size:
                store_time_dim_size = xr.open_zarr(zarr_path, consolidated=True).sizes[time_dim]
            else:
                store_time_dim_size = max(store_time_dim_size, ds.sizes[time_dim])

            atleast_one_success = True if success else atleast_one_success

        del ds
        if (i > 0) and (i % GC_INTERVAL == 0):
            loguru.logger.debug(f"Forcing garbage collection after {i} processed shots")
            jax.clear_caches()
            gc.collect()

    # Exit if no datasets were successfully processed.
    if not atleast_one_success:
        loguru.logger.warning(f"No successful datasets were processed. Returning an empty dataset for {zarr_path}.")
        if extend_existing and os.path.exists(zarr_path):
            return xr.open_zarr(zarr_path, consolidated=None)  # Allow for both consolidated and non-consolidated zarr stores.
        else:
            return xr.Dataset()

    # Now that the store has been built, we can consolidate the metadata and rechunk if necessary.
    loguru.logger.info(f"Successfully processed at least some of the files. Chunking and consolidating metadata for {zarr_path}.")
    ds = xr.open_zarr(zarr_path, consolidated=True)

    # If episodes were padded to upper-bound dim_sizes, drop the slack now. This is only done
    # when a rechunking pass will rewrite the store anyway, so the trim is effectively free.
    if dim_sizes is not None and (mb_per_chunk is not None or episodes_per_chunk is not None):
        for dim in dim_sizes:
            if dim in ds.dims and dim != episode_dim:
                ds = trim_trailing_nan_slices(ds, dim)

    if mb_per_chunk is not None:
        # Compute the number of episodes per chunk based on the average size of per-episode datasets.
        n_episodes = ds.sizes[episode_dim]
        bytes_per_episode = ds.isel({episode_dim: 0}).nbytes
        mean_mb_per_episode = bytes_per_episode / (1024 * 1024)
        episodes_per_chunk = min(max(1, int(mb_per_chunk / mean_mb_per_episode)), n_episodes)
        loguru.logger.info(
            f"Determined {episodes_per_chunk} episodes per chunk based on provided mb_per_chunk={mb_per_chunk} and the computed mean_mb_per_episode={mean_mb_per_episode:.2f}."
        )

    if episodes_per_chunk is not None:
        loguru.logger.info(f"Chunking the dataset with {episodes_per_chunk} episodes per chunk.")

        # We essentially only want chunking across episodes. So other dimensions should be covered by a single chunk.
        chunk_spec = {episode_dim: episodes_per_chunk} | {k: ds.sizes[k] for k in ds.dims if k != episode_dim}

        ds = zarr_chunk(ds, chunk_spec=chunk_spec)

        # Save the chunked dataset to a temporary path and then rename it to the final zarr path.
        tmp_path = zarr_path + ".tmp"
        ds.to_zarr(tmp_path, mode="w", consolidated=True)
        shutil.rmtree(zarr_path)  # Remove the old zarr store if it exists.
        os.rename(tmp_path, zarr_path)

    loguru.logger.info(f"Successfully processed at least some of the files. Consolidating metadata for {zarr_path}.")
    zarr.consolidate_metadata(zarr_path)
    return xr.open_zarr(zarr_path, consolidated=True)


def trim_trailing_nan_slices(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """Drop trailing slices along dim that are NaN across all variables and all other dimensions.

    Used to remove slack left over when episodes were padded to an upper-bound size.
    """
    slice_has_data = None
    for var in ds.data_vars:
        if dim not in ds[var].dims:
            continue
        other_dims = [d for d in ds[var].dims if d != dim]
        var_has_data = ds[var].notnull().any(other_dims)
        slice_has_data = var_has_data if slice_has_data is None else (slice_has_data | var_has_data)

    if slice_has_data is None:
        return ds

    valid = slice_has_data.compute().values
    if not valid.any():
        return ds

    last_valid = int(np.nonzero(valid)[0][-1]) + 1
    if last_valid < ds.sizes[dim]:
        loguru.logger.info(f"Trimming {ds.sizes[dim] - last_valid} trailing all-NaN slices along dimension {dim}.")
        ds = ds.isel({dim: slice(0, last_valid)})
    return ds


def extend_zarr_along_dim(zarr_path: os.PathLike, dim: str, n_extend: int) -> None:
    """Extend an existing zarr store along a specified dimension by padding it with nans."""
    ds = xr.open_zarr(zarr_path, consolidated=True)
    ds = ds.pad({dim: (0, n_extend)})
    ds_padding = ds.isel({dim: slice(-n_extend, None)})
    ds_padding = ds_padding.compute()  # Pull dataset into memory to avoid dask/chunking issues
    ds_padding.to_zarr(zarr_path, mode="a-", append_dim=dim, consolidated=True, align_chunks=True)


def add_to_zarr_store(  # noqa: PLR0912
    ds: xr.Dataset,
    zarr_path: os.PathLike,
    time_dim: str,
    episode_dim: str,
    store_time_dim_size: int | None = None,
    dim_sizes: dict[str, int] | None = None,
) -> bool:
    """Helper function to add a single xarray Dataset to a zarr store. Requires that the zarr store either doesn't exist or already contains all the dimensions in the provided Dataset.

    If dim_sizes is provided, the dataset is padded with NaNs to those sizes before being written. When every episode is padded this way (including the first), the store never needs to be extended, which avoids rewriting chunks for all previously written episodes whenever a later episode is larger.
    """

    # We need to reset all the non-index coordinates to make sure they get vary across episodes.
    ds = ds.reset_coords()

    if time_dim in ds.coords:
        ds = ds.drop_vars(time_dim)

    # If the episode dimension is not present, expand the dataset to include it.
    if episode_dim not in ds.dims:
        ds = ds.set_coords(episode_dim)
        ds = ds.expand_dims(episode_dim)

    if ds.sizes[episode_dim] > 1:
        raise ValueError(
            f"The dataset has more than one episode ({ds.sizes[episode_dim]}). "
            f"Please ensure that the dataset is for a single episode before adding it to the Zarr store."
        )

    # If only some of the variables have the episode dimension, we can't just expand the whole dataset.
    # We need to ensure that all variables have the episode dimension.
    for var in ds.data_vars:
        if episode_dim not in ds[var].dims:
            # If the variable does not have the episode dimension, we need to add it.
            ds[var] = ds[var].expand_dims(episode_dim)

    # Pad the dataset up to the provided upper-bound sizes so the store never needs extending.
    if dim_sizes:
        pad_to_bound = {}
        for dim, target_size in dim_sizes.items():
            if dim not in ds.dims or dim == episode_dim:
                continue
            if ds.sizes[dim] > target_size:
                raise ValueError(
                    f"Dimension {dim} has size {ds.sizes[dim]}, which exceeds the dim_sizes upper bound of {target_size}. "
                    f"Please ensure dim_sizes covers the largest episode."
                )
            if ds.sizes[dim] < target_size:
                pad_to_bound[dim] = (0, target_size - ds.sizes[dim])
        if pad_to_bound:
            ds = ds.pad(pad_to_bound)

    if not os.path.exists(zarr_path):
        loguru.logger.info(f"Zarr store at {zarr_path} does not exist. Creating a new one.")
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
        return True
    else:
        ds_store = xr.open_zarr(zarr_path, consolidated=True)

        # Appending a variable the store doesn't have creates it with a single episode,
        # leaving the store with conflicting episode-dim sizes that break every later open.
        store_vars = set(ds_store.data_vars)
        ds_vars = set(ds.data_vars)
        if ds_vars != store_vars:
            raise ValueError(
                f"Variable mismatch between dataset and existing zarr store: "
                f"missing from dataset {sorted(store_vars - ds_vars)}, "
                f"new in dataset {sorted(ds_vars - store_vars)}. "
                f"All episodes must provide the same set of variables."
            )

        # Handle dimension size changes for all dimensions except episode_dim
        pad_dims = {}
        extend_dims = {}

        # Check all other dimensions that exist in both datasets
        considered_dims = set(ds.dims) - {episode_dim}
        for dim in considered_dims:
            if dim in ds_store.dims:
                if dim == time_dim and store_time_dim_size is not None:
                    store_dim_size = store_time_dim_size
                else:
                    store_dim_size = ds_store.sizes[dim]

                ds_dim_size = ds.sizes[dim]

                if ds_dim_size < store_dim_size:
                    pad_dims[dim] = (0, store_dim_size - ds_dim_size)
                elif ds_dim_size > store_dim_size:
                    extend_dims[dim] = ds_dim_size - store_dim_size
            else:
                raise ValueError(f"Dimension {dim} is not present in the zarr store.")

        # Extend the zarr store for any dimensions that need to grow
        for dim, n_extend in extend_dims.items():
            extend_zarr_along_dim(zarr_path, dim, n_extend)

        # Pad the dataset for any dimensions that are too small
        if pad_dims:
            ds = ds.pad(pad_dims)

        # Append the new dataset to the existing zarr store.
        # a- means we append only to variables that have episode_dim.
        ds.to_zarr(zarr_path, mode="a-", append_dim=episode_dim, consolidated=True)
        # According to xarray docs, stale consolidated metadata can cause issues.
        # Should re-consolidate between each addition.
        zarr.consolidate_metadata(zarr_path)
        return True


def zarr_chunk(ds: xr.Dataset, chunk_spec: dict) -> xr.Dataset:
    """Chunk the dataset according to the provided chunk specification.
    For some reason we need to delete the chunks encoding from the dataset before saving it to zarr.
    See the stackoverflow issue:
        https://stackoverflow.com/questions/67476513/zarr-not-respecting-chunk-size-from-xarray-and-reverting-to-original-chunk-size
    """
    ds = ds.chunk(chunk_spec)
    for var in ds.data_vars:
        if "chunks" in ds[var].encoding:
            del ds[var].encoding["chunks"]
    for coord_name in ds.coords:  # Also check coords if they are being saved as arrays
        if hasattr(ds[coord_name], "encoding") and "chunks" in ds[coord_name].encoding:
            del ds[coord_name].encoding["chunks"]
    return ds
