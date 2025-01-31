import typing
import warnings

import jax.numpy as jnp
import jax_dataloader as jdl
import xarray as xr
import xbatcher
from jax_dataloader.loaders.jax import DataLoaderJAX
from jaxtyping import Array

from popsim.array_utils import contiguous_true_end_of_axis_mask
from popsim.ml._types import TrainingMetadata
from popsim.ml.preprocess_utils import shift_time_to_not_nan

DEFAULT_SAMPLE_DIM = "sample"


class DataLoader:
    """A thin wrapper around a jax_dataloader.DataLoader with convenience properties."""

    dl: DataLoaderJAX

    def __init__(self, dl: DataLoaderJAX):
        self.dl = dl
        # Check that training metadata is set.
        if dl.dataset.ds.popsim_ml.training_metadata is None:
            raise ValueError("Training metadata must be set in the dataset.")

    def __len__(self):
        return len(self.dl)

    def __next__(self):
        return next(self.dl)

    def __iter__(self):
        return iter(self.dl)

    @property
    def ds(self) -> xr.Dataset:
        return self.dl.dataset.ds


class XarrayPreppedDataset(jdl.Dataset):
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


def make_standard_dataloaders(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    input_vars: list[str],
    target_vars: list[str],
    split_fracs: typing.Sequence[float],
    key: int,
    extra_vars: typing.Optional[list[str]] = None,
    state_init_vars: typing.Optional[list[str]] = None,
    batch_size: typing.Optional[int] = None,
    segment_length: typing.Optional[int] = None,
    segment_overlap: typing.Optional[int] = 0,
) -> typing.Sequence[DataLoader]:
    if state_init_vars is None and segment_length is not None or segment_overlap != 0:
        raise ValueError("segment_length and segment_overlap are only valid when state_init_vars are provided")
    if state_init_vars is None:

        def dl_fun(ds_, shuffle):
            return make_time_indep_dataloader(
                ds=ds_,
                time_coord=time_coord,
                episode_coord=episode_coord,
                input_vars=input_vars,
                target_vars=target_vars,
                extra_vars=extra_vars,
                batch_size=batch_size,
                shuffle=shuffle,
            )
    else:

        def dl_fun(ds_, shuffle):
            return make_dataloader(
                ds=ds_,
                time_coord=time_coord,
                episode_coord=episode_coord,
                state_init_vars=state_init_vars,
                input_vars=input_vars,
                target_vars=target_vars,
                extra_vars=extra_vars,
                segment_length=segment_length,
                segment_overlap=segment_overlap,
                batch_size=batch_size,
                shuffle=shuffle,
            )

    episode_var_dim, _ = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)
    datasets = ds.popsim_ml.split_along_dim(episode_var_dim, split_fracs, key)

    # By default, only shuffle the first dataset.
    shuffle = (True if i == 0 else False for i in range(len(datasets)))
    return [dl_fun(ds_, sh) for ds_, sh in zip(datasets, shuffle)]


def make_time_indep_dataloader(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    input_vars: list[str],
    target_vars: list[str],
    extra_vars: typing.Optional[list[str]] = None,
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
        extra_vars (list[str], optional): Names of additional variables to include in the dataset. Defaults to None.
        batch_size (int, optional): Number of samples in each batch. If None, load all samples in a single batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.

    Returns:
        DataLoader: DataLoader for training the model wrapping a XarrayPreppedDataset.
    """
    ds = ds[input_vars + target_vars + (extra_vars or [])]
    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)
    sample_ds = ds.stack({DEFAULT_SAMPLE_DIM: (episode_var_dim, time_var_dim)}).dropna(DEFAULT_SAMPLE_DIM)
    sample_ds = sample_ds.transpose(DEFAULT_SAMPLE_DIM, ...)

    if batch_size is None:
        batch_size = len(sample_ds[DEFAULT_SAMPLE_DIM])

    train_meta = TrainingMetadata(
        sample_coord=DEFAULT_SAMPLE_DIM,
        sample_dim=DEFAULT_SAMPLE_DIM,
        input_vars=input_vars,
        target_vars=target_vars,
        episode_coord=episode_coord,
        episode_dim=episode_var_dim,
        time_dep_metadata=None,
    )

    sample_ds.popsim_ml.training_metadata = train_meta

    nan_report, nans_found = sample_ds.popsim_ml.generate_nan_report()

    if nans_found:
        warnings.warn(f"NaNs found in dataset. NaN report: \n{nan_report}", stacklevel=2)

    dl = DataLoader(
        DataLoaderJAX(
            XarrayPreppedDataset(ds=sample_ds),
            backend="jax",
            batch_size=batch_size,
            shuffle=shuffle,
        )
    )
    return dl


def make_dataloader(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    state_init_vars: list[str],
    input_vars: list[str],
    target_vars: list[str],
    extra_vars: typing.Optional[list[str]] = None,
    segment_length: typing.Optional[int] = None,
    segment_overlap: int = 0,
    batch_size: typing.Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """Given a multi-episode time series dataset, generate a DataLoader. This function does some pre-processing, and you should expect the resultant data to have the following properties:
        1. The data is segmented into samples of length `segment_length` with `segment_overlap` overlap.
        2. Incomplete samples at the end of the episode have their data and times forward-filled.
        3. Samples where inputs and targets are all NaN are dropped.

    Args:
        ds (xr.Dataset): Input dataset.
        time_coord (str): Name of the time coordinate variable.
        episode_coord (str): Name of the episode coordinate variable (e.g. "shot" or "simulation").
        state_init_vars (list[str]): Names of the variables required to initialize the state of the module.
        input_vars (list[str]): Names of the variables to be fed into the "Inputs" structure of the module.
        target_vars (list[str]): Names of the target variables that the module predicts.
        extra_vars (list[str], optional): Names of additional variables to include in the dataset. Defaults to None.
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

    model_vars = state_init_vars + input_vars
    ds = ds[model_vars + target_vars + (extra_vars or [])]
    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)

    ds = shift_time_to_not_nan(ds, episode_dim=episode_var_dim, time_coord=time_coord, how="any", subset=model_vars)

    # Construct the model_dims dictionary to define the input dimension the model will see.
    # We want the model to see a fixed number of time steps and data from a single episode.
    # However, we want to keep all other dimensions the same.
    non_episode_time_dims = {k: v for k, v in ds.sizes.items() if k not in [episode_var_dim, time_var_dim]}

    if segment_length is None:
        # If the segment length is not specified, use the full episode as the segment.
        segment_length = ds.sizes[time_var_dim]

    model_dims = {time_var_dim: segment_length} | non_episode_time_dims

    # Use a first call to Xbatcher to segment the episodes to the desired length and create a dataset with dimensions
    # (sample, time, other_dims).
    sample_ds = next(
        iter(xbatcher.BatchGenerator(ds, input_dims=model_dims, input_overlap={time_var_dim: segment_overlap}, concat_input_dims=True))
    )

    # By convention, the BatchGenerator adds "_input" to the time dimension.
    time_dim_sample_ds = f"{time_var_dim}_input"

    # Drop samples where the data is all NaN.
    sample_ds = sample_ds.dropna(DEFAULT_SAMPLE_DIM, how="all", subset=input_vars + target_vars)

    # Squeeze the sample_ds to get rid of extraneous dimensions.
    # For example, when "segment_length=1", we get rid of the "time" dimension.
    sample_ds = sample_ds.squeeze()

    # Forward fill the end of each sample along the time dimension to handle segments with unequal lengths.
    # This is distinct from forward-filling the whole dataset because this set of nans are artificially created via the segmenting process.
    sample_ds = ffill_end_of_time_padding(sample_ds, time_coord, time_dim_sample_ds)

    if batch_size is None:
        batch_size = len(sample_ds[DEFAULT_SAMPLE_DIM])

    train_meta = TrainingMetadata(
        sample_coord=DEFAULT_SAMPLE_DIM,
        sample_dim=DEFAULT_SAMPLE_DIM,
        input_vars=input_vars,
        target_vars=target_vars,
        episode_coord=episode_coord,
        episode_dim=episode_var_dim,
        time_dep_metadata=TrainingMetadata.TimeDepMetadata(
            state_init_vars=state_init_vars, time_coord=time_coord, time_dim=time_dim_sample_ds
        ),
    )

    sample_ds.popsim_ml.training_metadata = train_meta

    nan_report, nans_found = sample_ds.popsim_ml.generate_nan_report()

    if nans_found:
        warnings.warn(f"NaNs found in dataset. They will be forward-filled. NaN report: \n{nan_report}", stacklevel=2)
        sample_ds = sample_ds.ffill(time_dim_sample_ds)

    dl = DataLoader(
        DataLoaderJAX(
            XarrayPreppedDataset(ds=sample_ds),
            backend="jax",
            batch_size=batch_size,
            shuffle=shuffle,
        )
    )
    return dl


def ffill_end_of_time_padding(ds: xr.Dataset, time_coord: str, time_dim: str) -> xr.Dataset:
    """The segmenting process results in nan-padding to handle segments with unequal lengths. This function forward-fills in the nan-padding at the end of the time dimension.

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
