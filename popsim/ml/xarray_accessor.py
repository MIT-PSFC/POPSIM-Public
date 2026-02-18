import typing

import tabulate
import xarray as xr

from popsim.ml.split_utils import split_dataset_by_fracs


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

    def split_along_dim(self, dim: str, fracs: typing.Sequence[float], key: int) -> typing.Sequence[xr.Dataset]:
        """Split the dataset along a dimension according to the provided fractions.

        Args:
            dim (str): dimension to split the dataset along.
            fracs (typing.Sequence[float]): fractions of splits to be produced.
            key (int): seed for pseudo-random number generation.

        Returns:
            typing.Sequence[xr.Dataset]: list of datasets split along the dimension.
        """
        return split_dataset_by_fracs(self._ds, fracs, dim, key)

    def generate_nan_report(self) -> tuple[str, bool]:
        """Generate a report of NaN values in the dataset.

        Returns:
            tuple[str, bool]: A report of NaN values and a boolean indicating if NaNs were found.
        """
        # Compute a boolean mask where NaNs are present
        nan_mask = self._ds.isnull()

        n_nans_total = sum([nan_mask[var].sum().item() for var in self._ds.data_vars])

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
