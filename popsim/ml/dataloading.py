import typing

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
import xbatcher
from jaxtyping import Array
from loguru import logger

from popsim.array_utils import contiguous_true_end_of_axis_mask
from popsim.ml._types import TrainingMetadata
from popsim.ml.envs import ModuleEvalEnvInput
from popsim.ml.preprocess_utils import expand_time_dim, shift_time_to_not_nan, trim_time_to_not_nan
from popsim.ml.utils import pad_time_with_epsilon_xr

DEFAULT_SAMPLE_DIM = "sample"
PRNG_KEY_VAR = "prng_key"


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
        input_vars = training_metadata.input_vars
        if PRNG_KEY_VAR in ds and PRNG_KEY_VAR not in input_vars:
            input_vars = [*input_vars, PRNG_KEY_VAR]
        if self.training_metadata.is_time_dependent:
            # Get the time and sample coordinate variable, then drop them from the dataset.
            # We want to drop them because having different coordinates will re-trigger JIT compilation.
            time = ds[training_metadata.time_dep_metadata.time_coord]
            samples = ds[training_metadata.sample_coord]
            sample_coords = [c for c in ds.coords if training_metadata.sample_dim in ds[c].dims]
            ds = ds.drop_vars(sample_coords)

            inputs = ds[input_vars].load()
            targets = ds[training_metadata.target_vars].load()

            # If the sample dimension is not in the time dimension (i.e. all samples have the same time base), expand time to include the sample dimension.
            if training_metadata.sample_dim not in time.dims:
                time = time.expand_dims({training_metadata.sample_dim: samples})

            # Grab the first time slice to get the initial state.
            state_init = (
                ds[training_metadata.time_dep_metadata.state_init_vars].isel({training_metadata.time_dep_metadata.time_dim: 0}).load()
            )

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
            inputs = ds[input_vars].load()
            targets = ds[training_metadata.target_vars].load()

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

    def update_prng_seed(self, key: int) -> "XarrayPreppedDataset":
        """Generate new random numbers for each sample in the dataset and add them as a new variable."""
        np.random.seed(key)
        random_numbers = np.random.randint(low=0, high=2**32, size=(self.n_samples,))
        if PRNG_KEY_VAR in self.ds:
            self.ds[PRNG_KEY_VAR].data = random_numbers
        else:
            self.ds[PRNG_KEY_VAR] = xr.DataArray(random_numbers, dims=[self.training_metadata.sample_dim])


class DataLoader:
    def __init__(
        self,
        dataset: XarrayPreppedDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        key: jax.random.PRNGKey = jax.random.key(0),  # noqa: B008
        generate_prng: bool = False,
        **kwargs,
    ):
        self.key = key
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generate_prng = generate_prng

        if self.generate_prng and not shuffle:
            raise ValueError("generate_prng must be False if shuffle is False")

    def __len__(self):
        complete_batches, remainder = divmod(len(self.indices), self.batch_size)
        return complete_batches if self.drop_last else complete_batches + bool(remainder)

    def __iter__(self):
        if self.generate_prng:
            # When shuffling, presume training mode.
            # Seed *must* be in [0, 2**32) for np.random.seed compatibility.
            # Also, in x32 mode Jax defaults to int32, so we use that here. This is plenty of seeds and keys in either mode.
            seed = jax.random.randint(self.next_key(), (), minval=0, maxval=jnp.iinfo(jnp.int32).max, dtype=jnp.int32).item()
            self.dataset.update_prng_seed(seed)

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
        lim_dl = DataLoader(prep_ds, self.batch_size, self.shuffle, self.drop_last, key=self.key)

        return lim_dl

    @property
    def indices(self):
        return np.arange(len(self.dataset))

    @property
    def ds(self) -> xr.Dataset:
        return self.dataset.ds

    @property
    def metrics(self) -> dict:
        out = {"n_samples": self.dataset.n_samples, "n_GB": self.dataset.ds.nbytes / 1e9}
        return out

    @property
    def metadata(self) -> TrainingMetadata:
        return self.dataset.training_metadata


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


def _drop_nan_and_repad(ds_orig, time_var_dim, how, subset):
    """Given a dataset composed of a SINGLE episode, drop time slices where NaNs are present according to the specified method,
    and then pad the end of the episode with NaNs to maintain the original length in the time dimension.
    Note: data is not dropped and padded independently between episodes, so this function should only be applied with ds.groupby(episode_dim).map()
    """
    original_size = ds_orig.sizes[time_var_dim]
    ds_cleaned = ds_orig.dropna(time_var_dim, how=how, subset=subset)
    new_size = ds_cleaned.sizes[time_var_dim]
    if new_size < original_size:
        n_pad = original_size - new_size
        ds_padded = ds_cleaned.pad({time_var_dim: (0, n_pad)}, constant_values=np.nan)
        return ds_padded
    else:
        return ds_cleaned


def make_standard_dataloaders(
    ds: xr.Dataset,
    time_coord: str,
    episode_coord: str,
    input_vars: list[str],
    target_vars: list[str],
    split_fracs: typing.Sequence[float] = (0.6, 0.2, 0.2),
    key: int = 42,
    convert_xr_to_jnp: bool = True,
    extra_vars: list[str] | None = None,
    state_init_vars: list[str] | None = None,
    batch_size: int | None = None,
    segment_length: int | None = None,
    segment_overlap: int | None = 0,
    nan_handling: str | None = "drop_slice_all",
) -> typing.Sequence[DataLoader]:
    """Create standard training, validation, and test DataLoaders from a single dataset by splitting along the episode dimension.

    Args:
        ds (xr.Dataset): Input dataset.
        time_coord (str): Name of the time coordinate variable.
        episode_coord (str): Name of the episode coordinate variable (e.g. "shot" or "simulation").
        input_vars (list[str]): Names of the input variables that go into the model.
        target_vars (list[str]): Names of the target variables that the model predicts.
        split_fracs (typing.Sequence[float], optional): Fractions to split the dataset into. Should sum to 1.0. Defaults to (0.6, 0.2, 0.2).
        key (int, optional): Random seed for shuffling the dataset before splitting. Defaults to 42.
        convert_xr_to_jnp (bool, optional): Whether to convert the xarray dataset to a dictionary of jnp arrays before loading the data into the model. Defaults to True.
        extra_vars (list[str], optional): Names of additional variables to include in the dataset.
        state_init_vars (list[str], optional): Names of the variables required to initialize the state of the module.
        batch_size (int, optional): Number of samples in each batch. If None, load all samples in a single batch.
        segment_length (int, optional): Number of time steps used in each training segment. If None, then treat the full episode as a segment.
        segment_overlap (int, optional): Number of time steps that each segment overlaps with the previous segment. Defaults to 0.
        nan_handling (str, optional): Passed to make_time_dep_dataloader, see that function for details. Defaults to "drop_slice_all".
    Returns:
        typing.Sequence[DataLoader]: List of DataLoaders for training, validation, and testing.
    """

    episode_var_dim, _ = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)
    datasets = ds.popsim_ml.split_along_dim(episode_var_dim, split_fracs, key)

    return make_dataloaders(
        datasets=datasets,
        time_coord=time_coord,
        episode_coord=episode_coord,
        input_vars=input_vars,
        target_vars=target_vars,
        convert_xr_to_jnp=convert_xr_to_jnp,
        extra_vars=extra_vars,
        state_init_vars=state_init_vars,
        batch_size=batch_size,
        segment_length=segment_length,
        segment_overlap=segment_overlap,
        shuffle=[True if i == 0 else False for i in range(len(datasets))],
        nan_handling=nan_handling,
    )


def make_dataloaders(
    datasets: list[xr.Dataset],
    time_coord: str,
    episode_coord: str,
    input_vars: list[str],
    target_vars: list[str],
    convert_xr_to_jnp: bool = True,
    extra_vars: list[str] | None = None,
    state_init_vars: list[str] | None = None,
    batch_size: int | None | typing.Sequence[int | None] = None,
    segment_length: int | None | typing.Sequence[int | None] = None,
    segment_overlap: int | None | typing.Sequence[int | None] = 0,
    shuffle: bool | None | typing.Sequence[bool] = None,
    nan_handling: str | None | typing.Sequence[str] = "drop_slice_all",
) -> typing.Sequence[DataLoader]:
    """Create DataLoaders from a list of datasets. Each dataset is processed independently to create a DataLoader.

    `batch_size`, `segment_length`, `segment_overlap`, `shuffle`, and `nan_handling` may be provided as sequences of the same size as the number of datasets,
    in which case each dataset will be processed with the corresponding value from the sequence.
    If they are provided as a single value, that value will be used for all datasets.
    If `shuffle` is not provided, the first dataset will be shuffled and the rest will not be, which is a common pattern for train/val/test splits.

    Args:
        datasets (list[xr.Dataset]): List of datasets to create DataLoaders from.
        time_coord (str): Name of the time coordinate variable.
        episode_coord (str): Name of the episode coordinate variable (e.g. "shot" or "simulation").
        input_vars (list[str]): Names of the input variables that go into the model.
        target_vars (list[str]): Names of the target variables that the model predicts.
        convert_xr_to_jnp (bool, optional): Whether to convert the xarray dataset to a dictionary of jnp arrays before loading the data into the model. Defaults to True.
        extra_vars (list[str], optional): Names of additional variables to include in the dataset.
        state_init_vars (list[str], optional): Names of the variables required to initialize the state of the module.
        batch_size (int, typing.Sequence[int], optional): Number of samples in each batch. If None, load all samples in a single batch.
        segment_length (int, typing.Sequence[int], optional): Number of time steps used in each training segment. If None, then treat the full episode as a segment.
        segment_overlap (int, typing.Sequence[int], optional): Number of time steps that each segment overlaps with the previous segment. Defaults to 0.
        shuffle (bool, typing.Sequence[bool], optional): Whether to shuffle the samples in each DataLoader. If None, only the first DataLoader is shuffled.
        nan_handling (str, typing.Sequence[str], optional): Passed to make_time_dep_dataloader, see that function for details. Defaults to "drop_slice_all".

    Returns:
        list[DataLoader]: List of DataLoaders created from the input datasets.
    """
    if state_init_vars is None and (segment_length is not None or segment_overlap != 0):
        raise ValueError(
            "segment_length and segment_overlap are only valid for making time-dependent dataloaders, when state_init_vars are provided"
        )

    def _check_and_format_args(batch_size, segment_length, segment_overlap, shuffle, nan_handling):  # noqa: PLR0912
        # Ensure all variables which may be passed in as a sequence are formatted as tuples of the same length as the number of datasets.
        if batch_size is None:
            batch_size = [None for _ in datasets]
        elif isinstance(batch_size, int):
            batch_size = [batch_size for _ in datasets]
        elif isinstance(batch_size, typing.Sequence):
            if len(batch_size) != len(datasets):
                raise ValueError(
                    f"If batch_size is provided as a list, it must be the same length as the number of datasets. Got {len(batch_size)} values for {len(datasets)} datasets."
                )
        else:
            raise ValueError(f"batch_size must be an int, a sequence of ints, or None. Got {type(batch_size)}")

        if segment_length is None:
            segment_length = [None for _ in datasets]
        elif isinstance(segment_length, int):
            segment_length = [segment_length for _ in datasets]
        elif isinstance(segment_length, typing.Sequence):
            if len(segment_length) != len(datasets):
                raise ValueError(
                    f"If segment_length is provided as a list, it must be the same length as the number of datasets. Got {len(segment_length)} values for {len(datasets)} datasets."
                )
        else:
            raise ValueError(f"segment_length must be an int, a sequence of ints, or None. Got {type(segment_length)}")

        if segment_overlap is None:
            segment_overlap = [0 for _ in datasets]
        elif isinstance(segment_overlap, int):
            segment_overlap = [segment_overlap for _ in datasets]
        elif isinstance(segment_overlap, typing.Sequence):
            if len(segment_overlap) != len(datasets):
                raise ValueError(
                    f"If segment_overlap is provided as a list, it must be the same length as the number of datasets. Got {len(segment_overlap)} values for {len(datasets)} datasets."
                )
        else:
            raise ValueError(f"segment_overlap must be an int, a sequence of ints, or None. Got {type(segment_overlap)}")

        # By default, only shuffle the first dataset.
        if shuffle is None:
            shuffle = [True if i == 0 else False for i in range(len(datasets))]
        elif isinstance(shuffle, bool):
            shuffle = [shuffle for _ in range(len(datasets))]
        elif isinstance(shuffle, typing.Sequence):
            if len(shuffle) != len(datasets):
                raise ValueError(
                    f"If shuffle is provided as a list, it must be the same length as the number of datasets. Got {len(shuffle)} values for {len(datasets)} datasets."
                )

        if isinstance(nan_handling, str):
            nan_handling = [nan_handling for _ in datasets]
        elif isinstance(nan_handling, typing.Sequence):
            if len(nan_handling) != len(datasets):
                raise ValueError(
                    f"If nan_handling is provided as a list, it must be the same length as the number of datasets. Got {len(nan_handling)} values for {len(datasets)} datasets."
                )
        else:
            raise ValueError(f"nan_handling must be a str, a sequence of str, or None. Got {type(nan_handling)}")

        return batch_size, segment_length, segment_overlap, shuffle, nan_handling

    batch_size, segment_length, segment_overlap, shuffle, nan_handling = _check_and_format_args(
        batch_size, segment_length, segment_overlap, shuffle, nan_handling
    )

    if state_init_vars is None:

        def dl_fun(ds_, batch_size, shuffle):
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

        return [dl_fun(ds_, batch_size, sh) for ds_, batch_size, sh in zip(datasets, batch_size, shuffle, strict=False)]
    else:

        def dl_fun(ds_, batch_size, segment_length, segment_overlap, shuffle, nan_handling):
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
                nan_handling=nan_handling,
            )

        return [
            dl_fun(ds_, batch_size, seg_len, seg_overlap, sh, nan_handling)
            for ds_, batch_size, seg_len, seg_overlap, sh, nan_handling in zip(
                datasets, batch_size, segment_length, segment_overlap, shuffle, nan_handling, strict=False
            )
        ]


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
    generate_prng: bool = False,
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
        generate_prng (bool, optional): Whether to generate a prng variable for each sample in the dataset. This is only valid if shuffle is True. Defaults to False.

    Returns:
        DataLoader: DataLoader for training the model wrapping a XarrayPreppedDataset.
    """
    if extra_vars is None:
        extra_vars = []

    ds = ds[input_vars + target_vars + extra_vars]
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
        logger.warning(f"NaNs found in dataset. NaN report: \n{nan_report}")

    # Load time coordinate into memory for consistency with time-dependent dataloader
    if sample_ds[time_coord].chunks is not None:
        sample_ds[time_coord] = sample_ds[time_coord].load()

    prepped_ds = XarrayPreppedDataset(
        ds=sample_ds,
        training_metadata=train_meta,
    )

    dl = DataLoader(dataset=prepped_ds, batch_size=batch_size, shuffle=shuffle, generate_prng=generate_prng)
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
    generate_prng: bool = False,
    nan_handling: str | None = "drop_slice_all",
) -> DataLoader:
    """Given a multi-episode time series dataset, generate a DataLoader. This function does some pre-processing, and you should expect the resultant data to have the following properties:
        1 (optionally). Time slices where any of the input/target/state/extra vars are NaN are dropped
        - How this is handled is determined by the `nan_handling` argument. "drop_slice_any" and "drop_slice_all" will remove time slices with any or all NaNs respectively (note that this will cause inconsistent timesteps), while "drop_segment" will keep all time slices but drop samples containing NaNs after segmenting (loses more data but maintains consistent timesteps).
        2. The data is segmented into samples of length `segment_length` with `segment_overlap` overlap.
        3. Incomplete samples at the end of the episode have their data and times forward-filled.
        4. Samples where input/target/state/extra vars contain NaN are dropped, or error is raised.

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
        generate_prng (bool, optional): Whether to generate a prng variable for each sample in the dataset. This is only valid if shuffle is True. Defaults to False.
        nan_handling (str | None, optional): Method for handling NaN values in the dataset. If "drop_slice_all", time slices where all of the vars are NaN are dropped, and if NaNs remain raises ValueError. If "drop_slice_any", time slices where any of the vars are NaN are dropped. If "drop_segment", samples containing NaNs are dropped. Defaults to "drop_slice_all".

    Returns:
        DataLoader: DataLoader for training the model wrapping a XarrayPreppedDataset.
    """
    if extra_vars is None:
        extra_vars = []

    assert time_coord in ds.coords, f"Time coordinate {time_coord} not found in dataset."
    assert episode_coord in ds.coords, f"Episode coordinate {episode_coord} not found in dataset."

    if segment_length is None and segment_overlap != 0:
        raise ValueError("segment_overlap should be 0 when segment_length is None.")

    if nan_handling not in ["drop_slice_all", "drop_slice_any", "drop_segment"]:
        raise ValueError(
            f"Invalid nan_handling method {nan_handling}. Must be one of 'drop_slice_all', 'drop_slice_any', or 'drop_segment'."
        )

    training_vars = set(input_vars + target_vars + state_init_vars + extra_vars)
    ds = ds[training_vars]
    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)

    # If dataset has 1D time, expand to 2D along the shot dimension.
    if episode_var_dim not in ds[time_coord].dims:
        ds, time_var_dim = expand_time_dim(ds, episode_var_dim, time_coord, f"{time_var_dim}_slice")

    if nan_handling == "drop_slice_all":
        # Drop time slices where all of the training vars are NaN. If any NaNs remain after this, an error will be raised later.
        ds = ds.groupby(episode_var_dim).map(lambda episode: _drop_nan_and_repad(episode, time_var_dim, how="all", subset=training_vars))
    elif nan_handling == "drop_slice_any":
        # Drop time slices where any of the training vars are NaN.
        ds = ds.groupby(episode_var_dim).map(lambda episode: _drop_nan_and_repad(episode, time_var_dim, how="any", subset=training_vars))

    # Shift each episode to remove time slices where any nan is present at the start
    ds = shift_time_to_not_nan(
        ds, episode_dim=episode_var_dim, time_coord=time_coord, time_dim=time_var_dim, how="any", subset=training_vars
    )
    # Set the time coordinate to NaN after the last non-NaN time slice in each episode
    ds = trim_time_to_not_nan(
        ds, episode_dim=episode_var_dim, time_coord=time_coord, time_dim=time_var_dim, how="any", subset=training_vars
    )
    # Reduce the size of the dataset if possible by dropping time slices where all training vars are NaN for all episodes.
    # No data will be affected by this operation, it just reduces the amount of padding needed later.
    ds = ds.dropna(time_var_dim, how="all", subset=training_vars)

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
        iter(
            xbatcher.BatchGenerator(
                ds, input_dims=model_dims, input_overlap={time_var_dim: segment_overlap}, concat_input_dims=True, preload_batch=False
            )
        )
    )

    # By convention, the BatchGenerator adds "_input" to the time dimension.
    time_dim_sample_ds = f"{time_var_dim}_input"

    # Drop samples made from the batching process where the data is all NaN.
    sample_ds = sample_ds.dropna(DEFAULT_SAMPLE_DIM, how="all", subset=training_vars)

    # Squeeze the sample_ds to get rid of extraneous dimensions.
    # For example, when "segment_length=1", we get rid of the "time" dimension.
    sample_ds = sample_ds.squeeze()

    # Forward fill the end of each sample along the time dimension to handle segments with unequal lengths.
    # This is distinct from forward-filling the whole dataset because this set of nans are artificially created via the segmenting process.
    sample_ds = ffill_end_of_time_padding(sample_ds, time_coord, time_dim_sample_ds)

    # This always needs to get run to ensure there are no NaNs in the resulting samples
    nan_report, nans_found = sample_ds.popsim_ml.generate_nan_report()
    if nans_found:
        if nan_handling == "drop_slice_all":
            raise ValueError(
                f"NaN values found in dataset after dropping time slices where all training vars are NaN.\n\
                NaN report: \n{nan_report}\n\
                Consider using a different nan_handling method or preprocessing the dataset to handle NaNs before passing to dataloader creation."
            )

        # Drop samples where any remaining data is NaN
        original_n_samples = sample_ds.sizes[DEFAULT_SAMPLE_DIM]
        sample_ds = sample_ds.dropna(DEFAULT_SAMPLE_DIM, how="any", subset=training_vars)
        n_dropped_samples = original_n_samples - sample_ds.sizes[DEFAULT_SAMPLE_DIM]
        logger.warning(
            f"Dropped {n_dropped_samples} out of {original_n_samples} samples with NaNs after segmenting and padding.\n\
            NaN report: \n{nan_report}\n\
            Consider preprocessing the dataset to handle NaNs before passing to dataloader creation.",
            stacklevel=2,
        )
        if sample_ds.sizes[DEFAULT_SAMPLE_DIM] == 0:
            raise ValueError("All samples were dropped due to NaN values. Cannot create DataLoader.")

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

    prepped_ds = XarrayPreppedDataset(
        ds=sample_ds,
        training_metadata=train_meta,
    )

    dl = DataLoader(
        dataset=prepped_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generate_prng=generate_prng,
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

    ds[time_coord] = pad_time_with_epsilon_xr(ds[time_coord].load(), time_dim)
    if DEFAULT_SAMPLE_DIM in ds[time_coord].dims:
        # Drop samples where time is all NaN.
        ds = ds.dropna(DEFAULT_SAMPLE_DIM, how="all", subset=[time_coord])

    return ds
