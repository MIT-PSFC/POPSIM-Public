import typing
from dataclasses import dataclass

import jax.numpy as jnp
import xarray as xr
import xbatcher
from jax_dataloader import DataLoader, Dataset
from jaxtyping import Array

from popsim.ml.preprocess_utils import shift_time_to_not_nan
from popsim.ml.types import EvalInput
from popsim.simulate import SimInput


def ds_to_dict_jnp(ds: xr.Dataset) -> dict[str, Array]:
    return {var: jnp.asarray(ds[var].values).squeeze() for var in ds.data_vars}


class XarrayPreppedDataset(Dataset):
    """Wraps a xr.Dataset that has been prepared for training a model and provides an interface to DataLoader."""

    @dataclass
    class TimeDepMetadata:
        state_init_vars: list[str]
        time_coord: str
        time_dim: str

    ds: xr.Dataset
    sample_coord: str
    sample_dim: str
    param_vars: list[str]
    target_vars: list[str]
    time_dep_metadata: typing.Optional[TimeDepMetadata] = None

    def __init__(
        self,
        ds: xr.Dataset,
        sample_coord: str,
        sample_dim: str,
        param_vars: list[str],
        target_vars: list[str],
        time_dep_metadata: typing.Optional[TimeDepMetadata] = None,
    ):
        self.ds = ds
        self.sample_coord = sample_coord
        self.sample_dim = sample_dim
        self.param_vars = param_vars
        self.target_vars = target_vars
        self.time_dep_metadata = time_dep_metadata

    def __len__(self):
        return self.ds[self.sample_dim].size

    def __getitem__(self, idx) -> "XarrayPreppedDataset":
        ds_slice = self.ds.isel({self.sample_dim: idx})
        return XarrayPreppedDataset(
            ds=ds_slice,
            sample_coord=self.sample_coord,
            sample_dim=self.sample_dim,
            param_vars=self.param_vars,
            target_vars=self.target_vars,
            time_dep_metadata=self.time_dep_metadata,
        )

    def __eq__(self, other: "XarrayPreppedDataset") -> bool:
        # xr.Dataset requires special handling for equality comparison.
        ds_equals = xr.testing.assert_identical(self.ds, other.ds)

        # Compare all other variables.
        all_else_equals = self.get_all_but_ds() == other.get_all_but_ds()
        return ds_equals and all_else_equals

    def __repr__(self):
        from pprint import pformat

        return pformat(vars(self), indent=4, width=1)

    @property
    def is_time_dependent(self) -> bool:
        """Check if the dataset is for a time-dependent training task.

        Returns:
            bool: True if the dataset is for a time-dependent training task.
        """
        return self.time_dep_metadata is not None

    def get_all_but_ds(self) -> dict[str, typing.Any]:
        """Return all attributes of the class except the dataset.

        Returns:
            dict[str, typing.Any]: Dictionary of all attributes except the dataset.
        """
        return {k: v for k, v in vars(self).items() if k != "ds"}

    def to_eval_input(self) -> EvalInput:
        if self.is_time_dependent:
            # Forward fill to replace missing time values with the last known time value.
            time = self.ds[self.time_dep_metadata.time_coord].ffill(dim=self.time_dep_metadata.time_dim).values

            # Grab the first time slice to get the initial state.
            state_init = ds_to_dict_jnp(self.ds[self.time_dep_metadata.state_init_vars].isel({self.time_dep_metadata.time_dim: 0}))
        else:
            time = None
            state_init = None
        sim_input = SimInput(
            time=time,
            initial_state=state_init,
            params=ds_to_dict_jnp(self.ds[self.param_vars]),
        )
        eval_input = EvalInput(
            sim_input=sim_input,
            targets=ds_to_dict_jnp(self.ds[self.target_vars]),
        )
        return eval_input


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

    assert episode_var_name in time_var_dims, f"{episode_coord} not in {time_var_dims}"
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
    sample_ds = ds.stack(sample=(episode_var_dim, time_var_dim)).dropna("sample", how="any")

    if batch_size is None:
        batch_size = len(sample_ds["sample"])

    dl = DataLoader(
        XarrayPreppedDataset(
            ds=sample_ds,
            sample_coord="sample",
            sample_dim="sample",
            param_vars=input_vars,
            target_vars=target_vars,
            time_dep_metadata=None,
        ),
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
    segment_length: int,
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
        segment_length (int): Number of time steps used in each training segment.
        segment_overlap (int): Number of time steps that each segment overlaps with the previous segment. Defaults to 0.
        batch_size (int, optional): Number of samples in each batch. If None, load all samples in a single batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.

    Returns:
        DataLoader: DataLoader for training the model wrapping a XarrayPreppedDataset.
    """

    input_vars = state_init_vars + param_vars
    ds = ds[input_vars + target_vars]
    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_coord, time_coord)

    ds = shift_time_to_not_nan(ds, episode_dim=episode_var_dim, time_coord=time_coord, how="any", subset=input_vars)

    # Construct the input_dims dictionary to define the input dimension the model will see.
    # We want the model to see a fixed number of time steps and data from a single episode.
    # However, we want to keep all other dimensions the same.
    non_episode_time_dims = {k: v for k, v in ds.sizes.items() if k not in [episode_var_dim, time_var_dim]}
    input_dims = {time_var_dim: segment_length} | non_episode_time_dims

    # Use a first call to Xbatcher to segment the episodes to the desired length and create a dataset with dimensions
    # (sample, time, other_dims).
    sample_ds = next(
        iter(xbatcher.BatchGenerator(ds, input_dims=input_dims, input_overlap={time_var_dim: segment_overlap}, concat_input_dims=True))
    )

    # By convention, the BatchGenerator adds "_input" to the time dimension.
    time_dim_sample_ds = f"{time_var_dim}_input"

    # Drop samples where the data is all NaN.
    sample_ds = sample_ds.dropna("sample", how="all", subset=input_vars + target_vars)

    # Squeeze the sample_ds to get rid of extraneous dimensions.
    # For example, when "segment_length=1", we get rid of the "time" dimension.
    sample_ds = sample_ds.squeeze()

    if batch_size is None:
        batch_size = len(sample_ds["sample"])

    dl = DataLoader(
        XarrayPreppedDataset(
            ds=sample_ds,
            sample_coord="sample",
            sample_dim="sample",
            param_vars=param_vars,
            target_vars=target_vars,
            time_dep_metadata=XarrayPreppedDataset.TimeDepMetadata(
                state_init_vars=state_init_vars, time_coord=time_coord, time_dim=time_dim_sample_ds
            ),
        ),
        backend="jax",
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dl
