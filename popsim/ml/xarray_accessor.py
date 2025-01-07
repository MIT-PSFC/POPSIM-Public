import typing

import jax.numpy as jnp
import tabulate
import xarray as xr
from jax_dataloader import DataLoader
from jaxtyping import Array

from popsim.ml._types import TrainingMetadata
from popsim.ml.dataloading import make_standard_dataloaders
from popsim.ml.envs import ModuleEvalEnvInput
from popsim.ml.split_utils import split_dataset_along_dim


def ds_to_dict_jnp(ds: xr.Dataset) -> dict[str, Array]:
    return {var: jnp.asarray(ds[var].values).squeeze() for var in ds.data_vars}


@xr.register_dataset_accessor("popsim_ml")
class PopsimMLAccessor:
    """Accessor for xarray Datasets to help with ML training tasks."""

    def __init__(self, ds: xr.Dataset):
        if not isinstance(ds, xr.Dataset):
            raise TypeError("PopsimMLAccessor can only be used with xarray Datasets.")
        self._ds = ds

    @property
    def ds(self) -> xr.Dataset:
        """Get the xarray Dataset."""
        return self._ds

    @property
    def training_metadata(self) -> TrainingMetadata:
        """Get training metadata."""
        meta = self._ds.attrs.get("training_metadata", None)
        if meta is None:
            raise ValueError("No training metadata found in the dataset. Was the dataset properly prepared?")
        return meta

    @training_metadata.setter
    def training_metadata(self, value: TrainingMetadata):
        """Set training metadata. Require that the object is a xr.Dataset."""
        self._ds.attrs["training_metadata"] = value

        # Check that the sample_coord is 1D
        sample_coord_name = value.sample_coord
        if sample_coord_name in self._ds:
            if self._ds[sample_coord_name].ndim != 1:
                raise ValueError(f"The sample coordinate '{sample_coord_name}' must be 1D.")

    @property
    def sample_dim(self) -> str:
        """Get the sample dimension."""
        sample_coord = self.sample_coord
        dims = sample_coord.dims
        # Check that it is 1D
        if len(dims) != 1:
            raise ValueError(f"The sample coordinate '{sample_coord}' must be 1D.")
        return dims[0]

    @property
    def sample_coord(self) -> str:
        """Get the sample coordinate."""
        sample_coord_name = self.training_metadata.sample_coord
        return self._ds[sample_coord_name]

    @property
    def n_samples(self) -> int:
        """Get the number of samples in the dataset."""
        return self.sample_coord.size

    def split_along_dim(self, dim: str, fracs: typing.Sequence[float], key: int) -> typing.Sequence[xr.Dataset]:
        """Split the dataset along a dimension according to the provided fractions.

        Args:
            dim (str): dimension to split the dataset along.
            fracs (typing.Sequence[float]): fractions of splits to be produced.
            key (int): seed for psuedo-random number generation.

        Returns:
            typing.Sequence[xr.Dataset]: list of datasets split along the dimension.
        """
        return split_dataset_along_dim(self._ds, fracs, dim, key)

    def generate_nan_report(self) -> tuple[str, bool]:
        """Generate a report of NaN values in the dataset.

        Returns:
            tuple[str, bool]: A report of NaN values and a boolean indicating if NaNs were found.
        """
        # Compute a boolean mask where NaNs are present
        nan_mask = self._ds.isnull()

        n_nans_total = nan_mask.to_array().sum().item()

        # If no NaNs are found, return early with False
        if n_nans_total == 0:
            return "", False

        table = []
        headers = ["Variable", "Number of NaNs", "NaNs as a Fraction of Total"]
        for var_name in self._ds.data_vars:
            var_data = self._ds[var_name]
            nan_count = var_data.isnull().sum().item()
            total_elements = var_data.size
            nan_fraction = nan_count / total_elements
            table.append([var_name, nan_count, nan_fraction])
        report = tabulate.tabulate(table, headers=headers, tablefmt="grid")
        # Return the report and True, indicating NaNs were found
        return report, True

    def prep_inputs_and_targets(self):
        """Prepare inputs and targets for training."""

        ds = self.ds
        params = ds_to_dict_jnp(ds[self.training_metadata.param_vars])
        targets = ds_to_dict_jnp(ds[self.training_metadata.target_vars])

        if not self.training_metadata.is_time_dependent:
            return params, targets

        # If the sample dimension is not in the time dimension, expand time to include the sample dimension.
        time = ds[self.training_metadata.time_dep_metadata.time_coord]
        if self.sample_dim not in time.dims:
            time = time.expand_dims({self.sample_dim: self.sample_coord})

        time = jnp.asarray(time.values)

        # Grab the first time slice to get the initial state.
        state_init = ds_to_dict_jnp(
            ds[self.training_metadata.time_dep_metadata.state_init_vars].isel({self.training_metadata.time_dep_metadata.time_dim: 0})
        )

        env_input = ModuleEvalEnvInput(
            initial_state=state_init,
            params=params,
            time=time,
        )

        return env_input, targets

    def make_dataloaders(
        self,
        time_coord: str,
        episode_coord: str,
        input_vars: list[str],
        target_vars: list[str],
        split_fracs: typing.Sequence[float],
        key: int,
        state_init_vars: typing.Optional[list[str]] = None,
        extra_vars: typing.Optional[list[str]] = None,
        batch_size: typing.Optional[int] = None,
        segment_length: typing.Optional[int] = None,
        segment_overlap: typing.Optional[int] = 0,
    ) -> tuple[DataLoader]:
        """Build dataloaders for training, validation, and testing, where the data will be split along the dimension corresponding to "episode_coord" according to the fractions provided by "split_fracs".

        To specify that the dataloader should be for a time-dependent model, provide the state_init_vars argument. If this argument is not provided, the dataloader will be for a time-independent model.

        By default, the first dataloader of the returned tuple will shuffle the data, while the others will not.

        Args:
            time_coord (str): name of the time coordinate.
            episode_coord (str): name of the episode coordinate.
            input_vars (list[str]): module input variables.
            target_vars (list[str]): module target variables.
            split_fracs (typing.Sequence[float]): fractions of splits to be produced. Must sum to 1.0. For example, [0.7, 0.15, 0.15] would split the data into 70% training, 15% validation, and 15% testing.
            key (int): seed for psuedo-random number generation for splitting the dataset.
            state_init_vars (typing.Optional[list[str]], optional): variables needed to initialize the state of the module, if it is time-dependent. Defaults to None.
            extra_vars (typing.Optional[list[str]], optional): additional variables to include in the dataloaders. Defaults to None.
            batch_size (typing.Optional[int], optional): sizes of batches for dataloading. If None, then full batches will be loaded. Defaults to None.
            segment_length (typing.Optional[int], optional): For time-dependent modules, specifies the length of training segments. Defaults to None.
            segment_overlap (typing.Optional[int], optional): For time-dependent modules, specifies how many time steps the segments overlap. Defaults to 0.

        Returns:
            tuple[DataLoader]: tuple of DataLoaders, where the first is for training (i.e. shuffled), and the subsequent one are for validation, testing, and any other purposes.
        """
        return make_standard_dataloaders(
            ds=self._ds,
            time_coord=time_coord,
            episode_coord=episode_coord,
            input_vars=input_vars,
            target_vars=target_vars,
            split_fracs=split_fracs,
            key=key,
            state_init_vars=state_init_vars,
            extra_vars=extra_vars,
            batch_size=batch_size,
            segment_length=segment_length,
            segment_overlap=segment_overlap,
        )
