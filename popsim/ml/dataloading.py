import typing

import jax.numpy as jnp
import xarray as xr
import xbatcher
from jax_dataloader import DataLoader, Dataset
from jaxtyping import Array

from popsim.array_utils import contiguous_true_end_of_axis_mask
from popsim.ml.preprocess_utils import shift_time_to_not_nan
from popsim.ml.xarray_accessor import TrainingMetadata

DEFAULT_SAMPLE_DIM = "sample"


class XarrayPreppedDataset(Dataset):
    """A thin wrapper around an xr.Dataset that implements the Dataset interface for jax_dataloader."""

    ds: xr.Dataset

    def __init__(
        self,
        ds: xr.Dataset,
    ):
        if ds.popsim_ml.training_metadata is None:
            raise ValueError("xr.Dataset must have training metadata to be used with XarrayPreppedDataset.")
        self.ds = ds

    def __len__(self):
        return self.ds.popsim_ml.n_samples

    def __getitem__(self, idx) -> "XarrayPreppedDataset":
        sample_dim = self.ds.popsim_ml.sample_dim
        ds_slice = self.ds.isel({sample_dim: idx})
        return XarrayPreppedDataset(ds_slice)

    def __eq__(self, other: "XarrayPreppedDataset") -> bool:
        # xr.Dataset requires special handling for equality comparison.
        ds_equals = self.ds.equals(other.ds)
        return ds_equals


def ds_to_dict_jnp(ds: xr.Dataset) -> dict[str, Array]:
    return {var: jnp.asarray(ds[var].values).squeeze() for var in ds.data_vars}


def _get_and_check_episode_and_time_dims(ds: xr.Dataset, episode_var_name: str, time_var_name: str) -> tuple[str, str]:
    """Given a dataset and the names of the episode and time variables, extract the dimension names of these variables and check that they are consistent.

    Args:
        ds (xr.Dataset): Dataset containing the episode and time variables.
        episode_var_name (str): Name of the episode variable.
        time_var_name (str): Name of the time variable.

    Returns:
        tuple[str, str]: The dimension names of the episode and time variables.
    """
    # Extract the dimension of the episode variable and make sure it is one-dimensional.
    episode_coord = ds[episode_var_name]
    episode_var_dims = list(episode_coord.sizes.keys())
    assert len(episode_var_dims) == 1, f"Expected one dimension for episode var {episode_coord}"
    episode_var_dim = episode_var_dims[0]

    # Extract the dimension of the time variable.
    time_coord = ds[time_var_name]
    time_var_dims = set(time_coord.sizes.keys())

    time_var_dims_minus_episode = [d for d in time_var_dims if d != episode_var_dim]

    assert (
        len(time_var_dims_minus_episode) == 1
    ), f"For time var {time_coord}, expected one dimension besides the episode dimension, got {time_var_dims_minus_episode}"
    time_var_dim = time_var_dims_minus_episode[0]
    return episode_var_dim, time_var_dim


def make_time_indep_dataloader(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    input_vars: list[str],
    target_vars: list[str],
    batch_size: typing.Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for training tasks that do not require time dependence.

    Args:
        ds (xr.Dataset): Input dataset.
        time_coord (str): Name of the time coordinate variable.
        episode_coord (str): Name of the episode coordinate variable (e.g. "shot" or "simulation").
        input_vars (list[str]): Names of the input variables that go into the model.
        target_vars (list[str]): Names of the target variables that the model predicts.
        batch_size (int, optional): Number of samples in each batch. If None, load all samples in a single batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.

    Returns:
        DataLoader: DataLoader for training the model wrapping a XarrayPreppedDataset.
    """
    ds = ds[input_vars + target_vars]
    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)
    sample_ds = ds.stack({DEFAULT_SAMPLE_DIM: (episode_var_dim, time_var_dim)}).dropna(DEFAULT_SAMPLE_DIM, how="any")

    if batch_size is None:
        batch_size = len(sample_ds[DEFAULT_SAMPLE_DIM])

    train_meta = TrainingMetadata(
        sample_coord=DEFAULT_SAMPLE_DIM,
        sample_dim=DEFAULT_SAMPLE_DIM,
        param_vars=input_vars,
        target_vars=target_vars,
        time_dep_metadata=None,
    )

    sample_ds.popsim_ml.training_metadata = train_meta

    dl = DataLoader(
        XarrayPreppedDataset(ds=sample_ds),
        backend="jax",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dl


def make_dataloader(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    state_init_vars: list[str],
    param_vars: list[str],
    target_vars: list[str],
    segment_length: typing.Optional[int] = None,
    segment_overlap: int = 0,
    batch_size: typing.Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Given a multi-episode time series dataset, generate a DataLoader.

    Args:
        ds (xr.Dataset): Input dataset.
        time_coord (str): Name of the time coordinate variable.
        episode_coord (str): Name of the episode coordinate variable (e.g. "shot" or "simulation").
        state_init_vars (list[str]): Names of the variables required to initialize the state of the module.
        param_vars (list[str]): Names of the variables to be fed into the "Params" structure of the module.
        target_vars (list[str]): Names of the target variables that the module predicts.
        segment_length (typing.Optional[int], optional): Number of time steps used in each training segment. If None, then treat the full episode as a segment. Defaults to None.
        segment_overlap (int): Number of time steps that each segment overlaps with the previous segment. Defaults to 0.
        batch_size (int, optional): Number of samples in each batch. If None, load all samples in a single batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.

    Returns:
        DataLoader: DataLoader for training the model wrapping a XarrayPreppedDataset.
    """

    assert time_coord in ds.coords, f"Time coordinate {time_coord} not found in dataset."
    assert episode_coord in ds.coords, f"Episode coordinate {episode_coord} not found in dataset."

    if segment_length is None and segment_overlap != 0:
        raise ValueError("segment_overlap should be 0 when segment_length is None.")

    input_vars = state_init_vars + param_vars
    ds = ds[input_vars + target_vars]
    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)

    ds = shift_time_to_not_nan(ds, episode_dim=episode_var_dim, time_coord=time_coord, how="any", subset=input_vars)

    # Construct the input_dims dictionary to define the input dimension the model will see.
    # We want the model to see a fixed number of time steps and data from a single episode.
    # However, we want to keep all other dimensions the same.
    non_episode_time_dims = {k: v for k, v in ds.sizes.items() if k not in [episode_var_dim, time_var_dim]}

    if segment_length is None:
        # If the segment length is not specified, use the full episode as the segment.
        segment_length = ds.sizes[time_var_dim]

    input_dims = {time_var_dim: segment_length} | non_episode_time_dims

    # Use a first call to Xbatcher to segment the episodes to the desired length and create a dataset with dimensions
    # (sample, time, other_dims).
    sample_ds = next(
        iter(xbatcher.BatchGenerator(ds, input_dims=input_dims, input_overlap={time_var_dim: segment_overlap}, concat_input_dims=True))
    )

    # By convention, the BatchGenerator adds "_input" to the time dimension.
    time_dim_sample_ds = f"{time_var_dim}_input"

    # Drop samples where the data is all NaN.
    sample_ds = sample_ds.dropna(DEFAULT_SAMPLE_DIM, how="all", subset=input_vars + target_vars)

    # Squeeze the sample_ds to get rid of extraneous dimensions.
    # For example, when "segment_length=1", we get rid of the "time" dimension.
    sample_ds = sample_ds.squeeze()

    # Forward fill the end of the time dimension to handle segments with unequal lengths.
    sample_ds = ffill_end_of_time_padding(sample_ds, time_coord, time_dim_sample_ds)

    if batch_size is None:
        batch_size = len(sample_ds[DEFAULT_SAMPLE_DIM])

    train_meta = TrainingMetadata(
        sample_coord=DEFAULT_SAMPLE_DIM,
        sample_dim=DEFAULT_SAMPLE_DIM,
        param_vars=param_vars,
        target_vars=target_vars,
        time_dep_metadata=TrainingMetadata.TimeDepMetadata(
            state_init_vars=state_init_vars, time_coord=time_coord, time_dim=time_dim_sample_ds
        ),
    )

    sample_ds.popsim_ml.training_metadata = train_meta

    dl = DataLoader(
        XarrayPreppedDataset(ds=sample_ds),
        backend="jax",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dl


def ffill_end_of_time_padding(ds: xr.Dataset, time_coord: str, time_dim: str) -> xr.Dataset:
    """The segmenting process results in nan-padding to handle segments with unequal lengths. This function fills in the nan-padding at the end of the time dimension.

    Args:
        ds (xr.Dataset): dataset with nan-padding at the end of the time dimension.
        time_coord (str): time coordinate variable.
        time_dim (str): time dimension.

    Returns:
        xr.Dataset: dataset with nan-padding at the end of the time dimension filled in.
    """
    # Get the axis of the time dimension
    time_axis = ds[time_coord].dims.index(time_dim)

    # Identify the elements that are end padding.
    padding_mask = contiguous_true_end_of_axis_mask(ds[time_coord].isnull().values, axis=time_axis)
    ds[time_coord] = xr.where(padding_mask, ds[time_coord].ffill(time_dim), ds[time_coord])

    ds = xr.where(padding_mask, ds.ffill(time_dim), ds)

    if DEFAULT_SAMPLE_DIM in ds[time_coord].dims:
        # Drop samples where time is all NaN.
        ds = ds.dropna(DEFAULT_SAMPLE_DIM, how="all", subset=[time_coord])

    return ds
