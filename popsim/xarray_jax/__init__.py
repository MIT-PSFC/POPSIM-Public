import jax
import xarray

from popsim.xarray_jax.types import DataArray, Dataset, Variable

jax.tree_util.register_pytree_node(xarray.Variable, lambda v: jax.tree.flatten(Variable.from_xarray(v)), jax.tree.unflatten)
jax.tree_util.register_pytree_node(xarray.IndexVariable, lambda v: jax.tree.flatten(Variable.from_xarray(v)), jax.tree.unflatten)
jax.tree_util.register_pytree_node(xarray.DataArray, lambda da: jax.tree.flatten(DataArray.from_xarray(da)), jax.tree.unflatten)
jax.tree_util.register_pytree_node(xarray.Dataset, lambda ds: jax.tree.flatten(Dataset.from_xarray(ds)), jax.tree.unflatten)

__all__ = ["DataArray", "Dataset", "Variable"]
