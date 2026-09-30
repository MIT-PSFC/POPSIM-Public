import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array

from popsim.ml.dataloading import DataLoader
from popsim.norm_data import ScalingType, apply_norm, apply_unnorm, norm_data_xr

VARIABLE_DIM = "variable"


def _row_dims(dl: DataLoader) -> list[str]:
    """The dims a loader spreads its rows over: the sample dim, plus the segment time dim of a time-dependent loader."""
    row_dims = [dl.metadata.sample_dim]
    if dl.metadata.is_time_dependent:
        row_dims.append(dl.metadata.time_dep_metadata.time_dim)
    return row_dims


class VariableFlattener(eqx.Module):
    """Normalize a set of variables and flatten them into one vector per sample.

    The normalization statistics have the feature shape of each variable, without sample or time dims.
    They broadcast against a single sample (*feature) and against a sequence of steps (time, *feature).
    Static metadata is stored as tuples aligned with `variables` so the module stays hashable under jit.
    """

    means: dict[str, Array]
    scales: dict[str, Array]
    variables: tuple[str, ...] = eqx.field(static=True)
    var_dims: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    var_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    @property
    def n_flat(self) -> int:
        return sum(math.prod(shape) for shape in self.var_shapes)

    @property
    def flat_slices(self) -> dict[str, tuple[int, int]]:
        """Start and stop of every variable inside the flattened vector."""
        slices = {}
        start = 0
        for name, shape in zip(self.variables, self.var_shapes, strict=True):
            stop = start + math.prod(shape)
            slices[name] = (start, stop)
            start = stop
        return slices

    def dims_of(self, name: str) -> tuple[str, ...]:
        return self.var_dims[self.variables.index(name)]

    def shape_of(self, name: str) -> tuple[int, ...]:
        return self.var_shapes[self.variables.index(name)]

    def _var_data(self, inputs: xr.Dataset | dict[str, Array], name: str) -> Array:
        """The array of one variable, checking its dims when given a Dataset. A leading time axis is allowed."""
        if isinstance(inputs, xr.Dataset):
            da = inputs[name]
            actual_dims = tuple(da.dims)
            expected_dims = self.dims_of(name)
            if actual_dims != expected_dims and actual_dims[1:] != expected_dims:
                raise ValueError(f"Variable {name} has dims {actual_dims}, expected {expected_dims} with an optional leading time axis.")
            return da.data
        return inputs[name]

    def normalize_flat(self, inputs: xr.Dataset | dict[str, Array]) -> Array:
        """Normalize every variable and concatenate them along the last axis.

        Axes in front of a variable's feature dims, such as time, are kept.
        Returns (n_flat,) for a single sample, or (n_time, n_flat) for a sequence of steps.
        """
        flat_parts = []
        for name, shape in zip(self.variables, self.var_shapes, strict=True):
            x = self._var_data(inputs, name)
            x_normed = apply_norm(x, self.means[name], self.scales[name])
            leading_shape = x_normed.shape[: x_normed.ndim - len(shape)]
            flat_parts.append(x_normed.reshape(*leading_shape, -1))
        return jnp.concatenate(flat_parts, axis=-1)

    def unflatten_unnorm(self, x_flat: Array) -> dict[str, Array]:
        """Split a flattened (normalized) vector back into per-variable arrays in physical units.

        A leading axis on x_flat (e.g. time) is kept in front of each variable's feature shape.
        """
        leading_shape = x_flat.shape[:-1]
        out = {}
        for name, shape in zip(self.variables, self.var_shapes, strict=True):
            start, stop = self.flat_slices[name]
            x_normed = x_flat[..., start:stop].reshape(*leading_shape, *shape)
            out[name] = apply_unnorm(x_normed, self.means[name], self.scales[name])
        return out

    def mean_per_variable(self, x_flat: Array) -> Array:
        """Mean of a flattened array over the features of each variable, shape (..., n_variables)."""
        var_means = [x_flat[..., start:stop].mean(axis=-1) for start, stop in self.flat_slices.values()]
        return jnp.stack(var_means, axis=-1)

    def normalized_rows(self, dl: DataLoader, n_max_rows: int, seed: int = 0) -> np.ndarray:
        """A (n_rows, n_flat) matrix of the normalized flattened samples of a loader, at most n_max_rows.

        Every step of a time-dependent loader is its own row.
        Samples are subsampled before anything is materialized, so large datasets stay cheap.
        """
        sample_dim, *step_dims = _row_dims(dl)
        n_steps = math.prod(dl.ds.sizes[d] for d in step_dims)
        n_samples = dl.ds.sizes[sample_dim]
        n_keep = min(n_samples, max(1, math.ceil(n_max_rows / n_steps)))
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(n_samples, size=n_keep, replace=False)
        keep_idx = np.sort(keep_idx)
        ds_kept = dl.ds[list(self.variables)].isel({sample_dim: keep_idx})

        rows = []
        for name in self.variables:
            da = ds_kept[name].transpose(sample_dim, *step_dims, *self.dims_of(name))
            x_normed = apply_norm(da.values, np.asarray(self.means[name]), np.asarray(self.scales[name]))
            rows.append(x_normed.reshape(n_keep * n_steps, -1))
        return np.concatenate(rows, axis=1)

    @classmethod
    def from_dataloader(
        cls, train_dl: DataLoader, variables: Sequence[str], scaling_type: ScalingType | str = ScalingType.QUANTILE_50
    ) -> "VariableFlattener":
        """Compute normalization statistics from a training DataLoader.

        Statistics are reduced over the sample dim, and over the segment time dim of a time-dependent loader,
        so each variable is normalized per feature.
        The loader's dataset must be loaded into memory.
        """
        variables = tuple(variables)
        ds = train_dl.ds[list(variables)]
        reduce_dims = _row_dims(train_dl)
        _, means_ds, scales_ds = norm_data_xr(ds, scaling_type, sample_dim=reduce_dims)

        means, scales, var_dims, var_shapes = {}, {}, [], []
        for name in variables:
            feature_dims = tuple(d for d in ds[name].dims if d not in reduce_dims)
            var_dims.append(feature_dims)
            var_shapes.append(tuple(ds.sizes[d] for d in feature_dims))
            means[name] = jnp.asarray(means_ds[name].transpose(*feature_dims).values, dtype=float)
            scales[name] = jnp.asarray(scales_ds[name].transpose(*feature_dims).values, dtype=float)

        return cls(means=means, scales=scales, variables=variables, var_dims=tuple(var_dims), var_shapes=tuple(var_shapes))
