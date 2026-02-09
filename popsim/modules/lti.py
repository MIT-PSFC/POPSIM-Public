import chex
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from popsim import TimeDepModule
from popsim.math_utils import eigen_decompose
from popsim.tree_util import convert_to_real


@chex.dataclass
class EigenReducedLTIData:
    Lambda: Array  # Eigenvalues
    V: Array  # Eigenvectors
    V_inv: Array  # Inverse of eigenvectors


class LTI(TimeDepModule):
    """A linear time-invariant (LTI) system module."""

    @chex.dataclass
    class Config:
        A: Array
        B: Array
        C: Array
        D: Array
        xdot0: Array | None = None
        y0: Array | None = None
        eigen_reduced_data: EigenReducedLTIData | None = None

    @chex.dataclass
    class State:
        x: Array

    @chex.dataclass
    class Params:
        u: Array

    @chex.dataclass
    class Output:
        y: Array
        debug: dict | None = None

    config: Config

    def __init__(self, config: Config):
        # Only allow real values in the configuration.
        self.config = config

    def __call__(self, state: State, params: Params) -> tuple[State, Output]:
        xdot0 = self.config.xdot0 if self.config.xdot0 is not None else jnp.zeros_like(state.x)
        y0 = self.config.y0 if self.config.y0 is not None else jnp.zeros_like(self.config.C @ state.x)

        debug = None

        if self.config.eigen_reduced_data is not None:
            # Check type of state.x, if it is not complex, raise an error
            if not jnp.iscomplexobj(state.x):
                raise ValueError("State x must be complex when using eigen-reduced LTI.")

            Lambda, V, V_inv = self.config.eigen_reduced_data.Lambda, self.config.eigen_reduced_data.V, self.config.eigen_reduced_data.V_inv

            # Project x modal space.
            z = V_inv @ state.x

            # Compute the dynamics in modal space.
            zdot = Lambda @ z + V_inv @ (self.config.B @ params.u + xdot0)

            # Project back to state space using the fact that x = V*z implies xdot = V*zdot.
            xdot = V @ zdot

            debug = {"z": z, "zdot": zdot}
        else:
            xdot = self.config.A @ state.x + self.config.B @ params.u + xdot0

        y = self.config.C @ state.x + self.config.D @ params.u + y0

        # We only want to evolve real-valued states and outputs.
        y = convert_to_real(y, check=False)

        return LTI.State(x=xdot), LTI.Output(y=y, debug=debug)

    def reduce_eigen(self, n_modes: int, max_eigenvalue: float | None = None) -> "LTI":
        if max_eigenvalue is None:
            max_eigenvalue = np.inf

        eigenvalues, V = eigen_decompose(self.config.A)

        # Only keep eigenvalues below the threshold
        keep = eigenvalues < max_eigenvalue
        eigenvalues = eigenvalues[keep]
        V = V[:, keep]

        # Truncate to n_mode modes
        eigenvalues_truncated = eigenvalues[:n_modes]
        V_truncated = V[:, :n_modes]

        # Construct truncated Λ and V_inv
        Lambda_truncated = np.diag(eigenvalues_truncated)
        V_inv_truncated = np.linalg.pinv(V_truncated)

        eigen_reduced_data = EigenReducedLTIData(Lambda=Lambda_truncated, V=V_truncated, V_inv=V_inv_truncated)

        new_config = self.config.replace(eigen_reduced_data=eigen_reduced_data)

        return LTI(new_config)
