import typing
import warnings
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


def check_and_print_nan_report(ds: xr.Dataset, sample_coord: str):
    # Compute a boolean mask where NaNs are present
    nan_mask = ds.isnull()

    n_nans = nan_mask.sum()

    if n_nans == 0:
        return

    # Reduce the mask along the 'time_slice_input' dimension to see if any NaNs exist in that dimension
    nan_any = nan_mask.any(dim="time_slice_input")

    table = []
    headers = ["Variable", f"Samples with NaNs in {sample_coord}"]
    for var_name in nan_any.data_vars:
        nan_da = nan_any[var_name]
        if sample_coord in nan_da.coords:
            nan_values = nan_da.values
            samples_with_nan = nan_da.coords[sample_coord].values[nan_values]
            samples_str = ", ".join(map(str, samples_with_nan))
            table.append([var_name, samples_str])
        else:
            table.append([var_name, f"Coordinate '{sample_coord}' not found"])
    report = tabulate(table, headers=headers, tablefmt="grid")
    warnings.warn(f"Found NaNs in the dataset:\n{report}", stacklevel=2)


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
        return self._ds.attrs.get("training_metadata", None)

    @training_metadata.setter
    def training_metadata(self, value: TrainingMetadata):
        """Set training metadata. Require that the object is a xr.Dataset."""
        self._ds.attrs["training_metadata"] = value

        # Check that the sample_coord is 1D
        sample_coord_name = value.sample_coord
        if sample_coord_name in self._ds:
            if self._ds[sample_coord_name].ndim != 1:
                raise ValueError(f"The sample coordinate '{sample_coord_name}' must be 1D.")
        check_and_print_nan_report(self._ds, sample_coord_name)

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
