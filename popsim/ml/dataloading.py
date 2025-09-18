import typing
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import xbatcher
from jaxtyping import Array

from popsim.array_utils import contiguous_true_end_of_axis_mask
from popsim.ml._types import TrainingMetadata
from popsim.ml.envs import ModuleEvalEnvInput
from popsim.ml.preprocess_utils import expand_time_dim, shift_time_to_not_nan
from popsim.ml.utils import pad_time_with_epsilon_xr

DEFAULT_SAMPLE_DIM = "sample"


class XarrayPreppedDataset:
    """A thin wrapper around an xr.Dataset that also contains necessary metadata for training models."""

    ds: xr.Dataset
    training_metadata: TrainingMetadata

    def __init__(self, ds: xr.Dataset, training_metadata: TrainingMetadata):
        if ds[training_metadata.sample_coord].ndim != 1:
            raise ValueError(f"The sample coordinate '{training_metadata.sample_coord}' must be 1D.")

        if next(iter(ds.dims)) != training_metadata.sample_dim:
            ds = ds.transpose(training_metadata.sample_dim, ...)
        self.ds = ds
        self.training_metadata = training_metadata

    def __len__(self):
        return self.sample_coord.size

    def __getitem__(self, idx) -> "XarrayPreppedDataset":
        sample_dim = self.training_metadata.sample_dim
        ds_slice = self.ds.isel({sample_dim: idx})
        return XarrayPreppedDataset(ds_slice, self.training_metadata)

    def __eq__(self, other: "XarrayPreppedDataset") -> bool:
        # xr.Dataset requires special handling for equality comparison.
        ds_equals = self.ds.equals(other.ds)
        return ds_equals

    def __hash__(self):
        # Hash based on the id of the dataset and the training metadata
        return hash((id(self.ds), id(self.training_metadata)))

    def get_inputs_and_targets(self):
        training_metadata, ds = self.training_metadata, self.ds
        if self.training_metadata.is_time_dependent:
            # Get the time and sample coordinate variable, then drop them from the dataset.
            # We want to drop them because having different coordinates will re-trigger JIT compilation.
            time = ds[training_metadata.time_dep_metadata.time_coord]
            samples = ds[training_metadata.sample_coord]
            sample_coords = [c for c in ds.coords if training_metadata.sample_dim in ds[c].dims]
            ds = ds.drop_vars(sample_coords)

            inputs = ds[training_metadata.input_vars]
            targets = ds[training_metadata.target_vars]

            # If the sample dimension is not in the time dimension (i.e. all samples have the same time base), expand time to include the sample dimension.
            if training_metadata.sample_dim not in time.dims:
                time = time.expand_dims({training_metadata.sample_dim: samples})

            # Grab the first time slice to get the initial state.
            state_init = ds[training_metadata.time_dep_metadata.state_init_vars].isel({training_metadata.time_dep_metadata.time_dim: 0})

            if training_metadata.convert_xr_to_jnp:
                state_init = ds_to_dict_jnp(state_init)
                inputs = ds_to_dict_jnp(inputs)
                targets = ds_to_dict_jnp(targets)

            env_input = ModuleEvalEnvInput(
                initial_state=state_init,
                inputs=inputs,
                time=time.data,
            )

            return env_input, targets
        else:
            # Time-independent case.
            sample_coords = [c for c in ds.coords if training_metadata.sample_dim in ds[c].dims]
            ds = ds.drop_vars(sample_coords)
            inputs = ds[training_metadata.input_vars]
            targets = ds[training_metadata.target_vars]

            if training_metadata.convert_xr_to_jnp:
                inputs = ds_to_dict_jnp(inputs)
                targets = ds_to_dict_jnp(targets)
            return inputs, targets

    @property
    def sample_coord(self) -> str:
        """Get the sample coordinate."""
        sample_coord_name = self.training_metadata.sample_coord
        return self.ds[sample_coord_name]

    @property
    def n_samples(self) -> int:
        """Get the number of samples in the dataset."""
        return self.sample_coord.size


class DataLoader:
    def __init__(
        self,
        dataset: XarrayPreppedDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        key: int = 0,
        **kwargs,
    ):
        self.key = jax.random.PRNGKey(key)
        self.dataset = dataset

        self.indices = np.arange(len(dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        complete_batches, remainder = divmod(len(self.indices), self.batch_size)
        return complete_batches if self.drop_last else complete_batches + bool(remainder)

    def __iter__(self):
        def EpochIterator(data, batch_size: int, indices: typing.Sequence[int]):
            for i in range(0, len(indices), batch_size):
                idx = indices[i : i + batch_size]
                yield data[idx]

        # shuffle (permutation) indices every epoch
        indices = jax.random.permutation(self.next_key(), self.indices).__array__() if self.shuffle else self.indices

        if self.drop_last:
            indices = indices[: len(self.indices) - len(self.indices) % self.batch_size]
        return EpochIterator(self.dataset, self.batch_size, indices)

    def next_key(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def limit_size(self, size: int, coord: str) -> "DataLoader":
        """Create a copy of the present DataLoader but with a restricted size along the specified coordinate.
        The resulting DataLoader's dataset has every entry corresponding to the first N unique values of this coordinate.

        Args:
            size (int): The size to limit the dataset to.
            coord (str): The coordinate along which to limit the dataset.

        Returns:
            DataLoader: A new DataLoader instance with the limited dataset.
        """

        if coord in self.ds.coords or coord in self.ds.dims:
            # Check if coord is a level in any MultiIndex
            multiindex_dim = None
            for dim_name, index in self.ds.indexes.items():
                if hasattr(index, "names") and coord in index.names:
                    multiindex_dim = dim_name
                    break

            # Get unique values of the coordinate and limit them
            coord_vals = np.unique(self.ds[coord].values)[:size]

            if multiindex_dim or coord not in self.ds.indexes:
                # Create boolean mask to handle multi-index or non-indexed coord
                mask = self.ds[coord].isin(coord_vals)
                lim_ds = self.ds.where(mask, drop=True)
            else:
                # Regular dimension or indexed coordinate case
                lim_ds = self.ds.sel({coord: coord_vals})
        else:
            raise ValueError(f"Coordinate '{coord}' not found in dataset dimensions or coordinates")

        prep_ds = XarrayPreppedDataset(lim_ds, self.dataset.training_metadata)
        # Extract integer from PRNG key if needed
        key_val = int(self.key[0]) if hasattr(self.key, "__getitem__") else self.key
        lim_dl = DataLoader(prep_ds, self.batch_size, self.shuffle, self.drop_last, key=key_val)

        return lim_dl

    @property
    def ds(self) -> xr.Dataset:
        return self.dataset.ds

    @property
    def metrics(self) -> dict:
        out = {"n_samples": self.dataset.n_samples, "n_GB": self.dataset.ds.nbytes / 1e9}
        return out


def ds_to_dict_jnp(ds: xr.Dataset) -> dict[str, Array]:
    return {var: jnp.asarray(ds[var].values) for var in ds.data_vars}


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

    assert len(time_var_dims_minus_episode) == 1, (
        f"For time var {time_coord}, expected one dimension besides the episode dimension, got {time_var_dims_minus_episode}"
    )
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
    convert_xr_to_jnp: bool = True,
    extra_vars: list[str] | None = None,
    state_init_vars: list[str] | None = None,
    batch_size: int | None = None,
    segment_length: int | None = None,
    segment_overlap: int | None = 0,
) -> typing.Sequence[DataLoader]:
    if (state_init_vars is None and segment_length is not None) or segment_overlap != 0:
        raise ValueError("segment_length and segment_overlap are only valid when state_init_vars are provided")
    if state_init_vars is None:

        def dl_fun(ds_, shuffle):
            return make_time_indep_dataloader(
                ds=ds_,
                time_coord=time_coord,
                episode_coord=episode_coord,
                input_vars=input_vars,
                target_vars=target_vars,
                convert_xr_to_jnp=convert_xr_to_jnp,
                extra_vars=extra_vars,
                batch_size=batch_size,
                shuffle=shuffle,
            )
    else:

        def dl_fun(ds_, shuffle):
            return make_time_dep_dataloader(
                ds=ds_,
                time_coord=time_coord,
                episode_coord=episode_coord,
                state_init_vars=state_init_vars,
                input_vars=input_vars,
                target_vars=target_vars,
                convert_xr_to_jnp=convert_xr_to_jnp,
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
    return [dl_fun(ds_, sh) for ds_, sh in zip(datasets, shuffle, strict=True)]


def make_time_indep_dataloader(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    input_vars: list[str],
    target_vars: list[str],
    convert_xr_to_jnp: bool = True,
    extra_vars: list[str] | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for training tasks that do not require time dependence.

    Args:
        ds (xr.Dataset): Input dataset.
        time_coord (str): Name of the time coordinate variable.
        episode_coord (str): Name of the episode coordinate variable (e.g. "shot" or "simulation").
        input_vars (list[str]): Names of the input variables that go into the model.
        target_vars (list[str]): Names of the target variables that the model predicts.
        convert_xr_to_jnp (bool, optional): Whether to convert the xarray dataset to a dictionary of jnp arrays before loading the data into the model. Defaults to True.
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
        convert_xr_to_jnp=convert_xr_to_jnp,
        time_dep_metadata=None,
    )
    nan_report, nans_found = sample_ds.popsim_ml.generate_nan_report()

    if nans_found:
        warnings.warn(f"NaNs found in dataset. NaN report: \n{nan_report}", stacklevel=2)

    prepped_ds = XarrayPreppedDataset(
        ds=sample_ds,
        training_metadata=train_meta,
    )

    dl = DataLoader(
        dataset=prepped_ds,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dl


def make_time_dep_dataloader(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    state_init_vars: list[str],
    input_vars: list[str],
    target_vars: list[str],
    convert_xr_to_jnp: bool = True,
    extra_vars: list[str] | None = None,
    segment_length: int | None = None,
    segment_overlap: int = 0,
    batch_size: int | None = None,
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
        convert_xr_to_jnp (bool, optional): Whether to convert the xarray dataset to a dictionary of jnp arrays before loading the data into the model. Defaults to True.
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

    # If dataset has 1D time, expand to 2D along the shot dimension.
    if episode_var_dim not in ds[time_coord].dims:
        ds, time_var_dim = expand_time_dim(ds, episode_var_dim, time_coord, f"{time_var_dim}_slice")

    ds = shift_time_to_not_nan(ds, episode_dim=episode_var_dim, time_coord=time_coord, time_dim=time_var_dim, how="any", subset=model_vars)

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
        convert_xr_to_jnp=convert_xr_to_jnp,
        time_dep_metadata=TrainingMetadata.TimeDepMetadata(
            state_init_vars=state_init_vars, time_coord=time_coord, time_dim=time_dim_sample_ds
        ),
    )

    nan_report, nans_found = sample_ds.popsim_ml.generate_nan_report()

    if nans_found:
        warnings.warn(f"NaNs found in dataset. They will be forward-filled. NaN report: \n{nan_report}", stacklevel=2)
        sample_ds = sample_ds.ffill(time_dim_sample_ds)

    prepped_ds = XarrayPreppedDataset(
        ds=sample_ds,
        training_metadata=train_meta,
    )

    dl = DataLoader(
        dataset=prepped_ds,
        batch_size=batch_size,
        shuffle=shuffle,
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

    padding_mask_da = xr.DataArray(padding_mask, dims=ds[time_coord].dims, coords=ds[time_coord].coords)

    # For some reason, if we don't reset the time coordinate, in some cases the time coordinate is dropped.
    # by the xr.where operation.
    ds = ds.reset_coords(time_coord)

    ds = xr.where(padding_mask_da, ds.ffill(time_dim), ds)

    ds[time_coord] = pad_time_with_epsilon_xr(ds[time_coord], time_dim)
    if DEFAULT_SAMPLE_DIM in ds[time_coord].dims:
        # Drop samples where time is all NaN.
        ds = ds.dropna(DEFAULT_SAMPLE_DIM, how="all", subset=[time_coord])

    return ds
