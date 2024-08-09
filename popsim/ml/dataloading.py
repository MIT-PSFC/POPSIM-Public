import typing

import xarray as xr
import xbatcher
from jax_dataloader import DataLoader, Dataset

from popsim.ml.preprocess_utils import shift_time_to_not_nan


class XBatcherDataset(Dataset):
    bgen: xbatcher.BatchGenerator

    def __init__(self, bgen: xbatcher.BatchGenerator):
        self.bgen = bgen

    def __len__(self):
        return len(self.bgen)

    def __getitem__(self, idx):
        idx = int(idx.squeeze())  # The Dataloader provides Jax arrays, but xbatcher expects ints.
        batch = self.bgen[idx].load()
        return batch


def make_dataloader(
    ds: xr.Dataset,
    time_var: str,
    episode_var: str,
    state_init_vars: list[str],
    param_vars: list[str],
    segment_length: int,
    segment_overlap: int,
    batch_size: typing.Optional[int] = None,
    shuffle: bool = True,
) -> xr.Dataset:
    """Given a multi-episode time series dataset, segment the data into training samples and create a DataLoader.

    Args:
        ds (xr.Dataset): input dataset.
        time_name (str): Name of the time variable.
        episode_name (str): Name of the episode variable.
        state_init_vars (list[str]): List of variables required to initialize the state of the model.
        param_vars (list[str]): List of variables that are time-varying parameters input to the model.
        segment_length (int): Number of time steps used in each training segment.
        segment_overlap (int): Number of time steps that each segment overlaps with the previous segment.
        batch_size (int, optional): Number of samples in each batch. If None, load all samples in a single batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.

    Returns:
        xr.Dataset: Dataset with dimensions (sample, time, other_dims).
    """

    def _get_and_check_episode_and_time_dims(ds: xr.Dataset, episode_var_name: str, time_var_name: str):
        # Extract the dimension of the episode variable and make sure it is one-dimensional.
        episode_var = ds[episode_var_name]
        episode_var_dims = list(episode_var.sizes.keys())
        assert len(episode_var_dims) == 1, f"Expected one dimension for episode var {episode_var}"
        episode_var_dim = episode_var_dims[0]

        # Extract the dimension of the time variable.
        time_var = ds[time_var_name]
        time_var_dims = set(time_var.sizes.keys())

        assert episode_var_name in time_var_dims, f"{episode_var} not in {time_var_dims}"
        time_var_dims_minus_episode = [d for d in time_var_dims if d != episode_var_dim]

        assert (
            len(time_var_dims_minus_episode) == 1
        ), f"For time var {time_var}, expected one dimension besides the episode dimension, got {time_var_dims_minus_episode}"
        time_var_dim = time_var_dims_minus_episode[0]
        return episode_var_dim, time_var_dim

    episode_var_dim, time_var_dim = _get_and_check_episode_and_time_dims(ds, episode_var, time_var)

    ds = shift_time_to_not_nan(ds, episode_dim=episode_var_dim, time_var=time_var, how="any", subset=state_init_vars)

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

    # Drop samples where the data is all NaN.
    sample_ds = sample_ds.dropna("sample", how="all", subset=state_init_vars + param_vars)

    if batch_size is None:
        batch_size = len(sample_ds["sample"])

    bgen = xbatcher.BatchGenerator(sample_ds, input_dims={"sample": batch_size})

    # batch_size = 1 because we are already batching in the xbatcher.BatchGenerator.
    # Telling the Dataloader batch_size=1 means "grab one batch at a time".
    dl = DataLoader(XBatcherDataset(bgen), backend="jax", batch_size=1, shuffle=shuffle)
    return dl
