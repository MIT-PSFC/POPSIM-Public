import os
import shutil
from collections.abc import Sequence
from typing import Any, Callable, Optional

import loguru
import xarray as xr
import zarr
from tqdm import tqdm


def build_tensorized_dataset(
    process_fn: Callable[[Any], xr.Dataset],
    identifiers: Sequence[Any],
    zarr_path: os.PathLike,
    time_dim: str,
    episode_dim: str,
    extend_existing: bool = False,
    episodes_per_chunk: Optional[int] = None,
    mb_per_chunk: Optional[int] = 100,
) -> xr.Dataset:
    """Build a tensorized multi-episode dataset that can then be used for training or evaluation.
    The user provides a function that processes data for a single episode, which returns an xarray Dataset for a single episode. This function will then build up the multi-episode dataset. This function was built with the intention of only needing to load a single episode at a time, allowing us to build up a dataset that is too large to fit in memory.

    Args:
        process_fn (Callable[[Any], xr.Dataset]): A function that takes in a episode identifier (e.g. a file path or a pulse number) and returns an xarray Dataset for a single episode.
        identifiers (Sequence[Any]): A sequence of iterables (e.g. file paths or pulse numbers) that will be processed by the process_fn to create the dataset.
        zarr_path (os.PathLike): path to the zarr store where the dataset will be saved.
        time_dim (str): The name of the time dimension in the dataset.
        episode_dim (str): The name of the episode dimension in the dataset.
        extend_existing (bool, optional): If zarr_path already exists and this is true, we will try to extend the existsing zarr_path. Defaults to False.
        episodes_per_chunk (Optional[int], optional): The number of episodes per chunk in storage. If this is not None, then this function will rechunk the built Zarr store once all the files are added. Defaults to 10.
        mb_per_chunk (Optional[int], optional): If specified, the resulting Zarr stored will be chunked along the episodes dimension with size max(1, int(mb_per_chunk / mean_mb_per_episode)). Defaults to None.

    Raises:
        ValueError: If zarr_path already exists and extend_existing is False, this function will raise an error.

    Returns:
        xr.Dataset: The xarray Dataset that was built up.
    """

    def _validate_inputs() -> None:
        if not zarr_path.endswith(".zarr"):
            raise ValueError(f"Provided zarr_path must end with .zarr, but got {zarr_path}")

        if os.path.exists(zarr_path) and extend_existing is False:
            raise ValueError(
                f"Zarr store at {zarr_path} already exists and extend_existing is False. Please remove it or set extend_existing to True."
            )
        if mb_per_chunk is not None and episodes_per_chunk is not None:
            raise ValueError("Please specify either mb_per_chunk or episodes_per_chunk, not both.")

    _validate_inputs()

    def get_and_preprocess(identifier: Any) -> Optional[xr.Dataset]:
        try:
            return process_fn(identifier)
        except Exception as e:
            # Warn the user.
            loguru.logger.warning(f"Error processing {identifier}: {e}")
            return None

    atleast_one_success = False  # Keep track of whether at least one dataset was successfully processed.
    store_time_dim_size = None  # Keep track of the size of the time dimension in the zarr store.

    # Iterate over the identifiers and process them one by one to build the dataset.
    bytes_per_ds = []
    for it in tqdm(identifiers, desc="Building the dataset..."):
        ds = get_and_preprocess(it)
        if ds is not None:
            bytes_per_ds.append(ds.nbytes)
            success = add_to_zarr_store(ds, zarr_path, time_dim, episode_dim, store_time_dim_size=store_time_dim_size)
            if success and not store_time_dim_size:
                store_time_dim_size = xr.open_zarr(zarr_path).sizes[time_dim]
            else:
                store_time_dim_size = max(store_time_dim_size, ds.sizes[time_dim])

            atleast_one_success = True if success else atleast_one_success

    if atleast_one_success:
        if mb_per_chunk is not None:
            loguru.logger.info(f"Successfully processed at least some of the files. Chunking and consolidating metadata for {zarr_path}.")
            ds = xr.open_zarr(zarr_path, consolidated=False)

            # Compute the number of episodes per chunk based on the average size of per-episode datasets.
            n_episodes = ds.sizes[episode_dim]
            mean_mb_per_episode = sum(bytes_per_ds) / len(bytes_per_ds) / (1024 * 1024)
            episodes_per_chunk = min(max(1, int(mb_per_chunk / mean_mb_per_episode)), n_episodes)

            # Chunk the dataset along the episode dimension.
            ds = zarr_chunk(ds, chunk_spec={episode_dim: episodes_per_chunk})

            # Save the chunked dataset to a temporary path and then rename it to the final zarr path.
            tmp_path = zarr_path + ".tmp"
            ds.to_zarr(tmp_path, mode="w", consolidated=True)
            shutil.rmtree(zarr_path)  # Remove the old zarr store if it exists.
            os.rename(tmp_path, zarr_path)
        else:
            loguru.logger.info(f"Successfully processed at least some of the files. Consolidating metadata for {zarr_path}.")
            zarr.consolidate_metadata(zarr_path)
        return xr.open_zarr(zarr_path, consolidated=True)
    else:
        loguru.logger.warning(f"No successful datasets were processed. Returning an empty dataset for {zarr_path}.")
        return xr.Dataset()


def extend_zarr_along_dim(zarr_path: os.PathLike, dim: str, n_extend: int) -> None:
    """Extend an existing zarr store along a specified dimension by padding it with nans."""
    ds = xr.open_zarr(zarr_path)
    ds = ds.pad({dim: (0, n_extend)})
    ds_padding = ds.isel({dim: slice(-n_extend, None)})
    ds_padding.to_zarr(zarr_path, mode="a-", append_dim=dim, consolidated=False)


def add_to_zarr_store(
    ds: xr.Dataset, zarr_path: os.PathLike, time_dim: str, episode_dim: str, store_time_dim_size: Optional[int] = None
) -> bool:
    """Helper function to add a single xarray Dataset to a zarr store."""

    # We need to reset all the non-index coordinates to make sure they get vary across episodes.
    ds = ds.reset_coords()

    # If there is still a coordinate with the same name as the time dimension, we need to reset it.
    if time_dim in ds.coords:
        ds = ds.reset_index(time_dim).reset_coords()

    # If the episode dimension is not present, expand the dataset to include it.
    if episode_dim not in ds.dims:
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

    if not os.path.exists(zarr_path):
        loguru.logger.info(f"Zarr store at {zarr_path} does not exist. Creating a new one.")
        ds.to_zarr(zarr_path, mode="w", consolidated=False)
        return True
    else:
        if store_time_dim_size is None:
            ds_store = xr.open_zarr(zarr_path)
            store_time_dim_size = ds_store.sizes[time_dim]
        if ds.sizes[time_dim] < store_time_dim_size:
            # If the new dataset is smaller than the store, we need to pad it before adding it to the store.
            ds = ds.pad({time_dim: (0, store_time_dim_size - ds.sizes[time_dim])})
        elif ds.sizes[time_dim] > store_time_dim_size:
            # We need to extend the existing zarr store along the time dimension.
            extend_zarr_along_dim(zarr_path, time_dim, ds.sizes[time_dim] - store_time_dim_size)
        # Append the new dataset to the existing zarr store.
        # a- means we append only to variables that have episode_dim.
        ds.to_zarr(zarr_path, mode="a-", append_dim=episode_dim, consolidated=False)
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
