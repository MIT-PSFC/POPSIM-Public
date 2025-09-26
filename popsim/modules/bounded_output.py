import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from popsim import TimeIndepModule
from popsim.math_utils import soft_clip


class BoundType(eqx.Enumeration):
    """Enumeration of supported bounding types for BoundedOutput module."""

    SOFT = "soft"
    CLIP = "clip"


class BoundedOutput(TimeIndepModule):
    """A module that wraps another module and bounds its output."""

    module: TimeIndepModule
    lower_bound: PyTree
    upper_bound: PyTree
    bound_type: BoundType = eqx.field(static=True, default_factory=lambda: BoundType.SOFT)

    def __init__(self, module: TimeIndepModule, lower_bound: Array, upper_bound: Array, bound_type: BoundType = BoundType.SOFT):
        self.module = module
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bound_type = bound_type

    def __call__(self, inputs: PyTree) -> Array:
        module_out = self.module(inputs)

        if self.bound_type == BoundType.CLIP:

            def bound_function(out, lb, ub):
                return jnp.clip(out, lb, ub)
        elif self.bound_type == BoundType.SOFT:

            def bound_function(out, lb, ub):
                return soft_clip(out, lb, ub)
        else:
            raise ValueError(f"Unsupported bound_type: {self.bound_type}. Supported types are {list(BoundType)}.")

        bounded_out = jax.tree.map(
            bound_function,
            module_out,
            self.lower_bound,
            self.upper_bound,
        )
        return bounded_out
