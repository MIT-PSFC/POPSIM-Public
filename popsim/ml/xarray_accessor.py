import typing
from dataclasses import dataclass

import jax.numpy as jnp
import tabulate
import xarray as xr
from jaxtyping import Array

from popsim.ml.envs import ModuleEvalEnvInput


def ds_to_dict_jnp(ds: xr.Dataset) -> dict[str, Array]:
    return {var: jnp.asarray(ds[var].values).squeeze() for var in ds.data_vars}


@dataclass
class TrainingMetadata:
    """Metadata for a training task."""

    @dataclass
    class TimeDepMetadata:
        state_init_vars: list[str]
        time_coord: str
        time_dim: str

    sample_coord: str
    sample_dim: str
    param_vars: list[str]
    target_vars: list[str]
    time_dep_metadata: typing.Optional[TimeDepMetadata] = None

    @property
    def is_time_dependent(self) -> bool:
        """Check if the metadata is for a time-dependent training task."""
        return self.time_dep_metadata is not None


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
