import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from popsim import TimeIndepModule
from popsim.norm_data import ScalingType, norm_data


class LinearAutoEncoder(TimeIndepModule):
    means: Array
    scaling: Array
    components: Array

    def forward(self, x: Array) -> Array:
        x = (x - self.means) / self.scaling
        projected = self.components_pseudoinv @ x
        return projected

    def backward(self, x: Array) -> Array:
        unprojected = self.components @ x
        out = unprojected * self.scaling + self.means
        return out

    @property
    def components_pseudoinv(self):
        return jnp.linalg.pinv(self.components)

    @classmethod
    def fit(cls, arr: Array, n_latent: int, scaling_type: ScalingType | str = ScalingType.STD) -> tuple["LinearAutoEncoder", Array]:
        """Initialize the linear autoencoder with SVD/PCA on the provided data.

        Args:
            arr (Array): The data array to fit the autoencoder on. Must be at most 2D. The first dimension is assumed to be the sample dimension.
            n_latent (int): The number of latent dimensions to reduce to.
            scaling_type (ScalingType | str, optional): The type of scaling to apply to the data before fitting. Defaults to ScalingType.STD.

        Raises:
            ValueError: If the input array has more than 2 dimensions.

        Returns:
            tuple[LinearAutoEncoder, Array]: The fitted LinearAutoEncoder instance and the explained variance ratio.
        """
        if arr.ndim > 2:
            raise ValueError(f"Input array must be at most 2D. Found array with shape {arr.shape}.")
        x, means, scaling_factor = norm_data(arr, scaling_type, sample_dim=0)

        n_samples, _ = x.shape

        # Factorize the data matrix with singular value decomposition.
        _U, S, Vt = jax.scipy.linalg.svd(x, full_matrices=False)

        # Compute the explained variance
        explained_variance = (S[:n_latent] ** 2) / (n_samples - 1)

        # Compute the total variance (sum of all singular values squared, scaled appropriately)
        total_variance = (S**2).sum() / (n_samples - 1)

        # Compute the explained variance ratio
        explained_variance_ratio = explained_variance / total_variance

        # Return the transformation matrix
        A = Vt[:n_latent]

        return cls(
            components=A.T,
            means=means.squeeze(),
            scaling=scaling_factor,
        ), explained_variance_ratio

    @classmethod
    def fit_flattened_tree(
        cls, tree: PyTree, n_latent: int, scaling_type: ScalingType | str = ScalingType.STD
    ) -> tuple["LinearAutoEncoder", Array]:
        """Flatten each sample of a tree into a 1D array and fit a LinearAutoEncoder to the resulting 2D array.

        Args:
            tree (PyTree): The input tree with arrays to fit the autoencoder on. Each array must be at most 2D. The first dimension of each array is assumed to be the sample dimension.
            n_latent (int): The number of latent dimensions to reduce to.
            scaling_type (ScalingType | str, optional): The type of scaling to apply to the data before fitting. Defaults to ScalingType.STD.

        Returns:
            tuple[LinearAutoEncoder, Array]: The fitted LinearAutoEncoder instance and the explained variance ratio.
        """

        def ravel_fn(x):
            # We only want the ravelled array, not the unflatten function.
            x_flat, _ = jax.flatten_util.ravel_pytree(x)
            return x_flat

        tree_flat = jax.vmap(ravel_fn)(tree)
        return LinearAutoEncoder.fit(tree_flat, n_latent, scaling_type)

    @staticmethod
    def fit_tree(tree: PyTree, n_latent: int | PyTree, scaling_type: ScalingType | str = ScalingType.STD) -> PyTree["LinearAutoEncoder"]:
        """Fit a LinearAutoEncoder to each leaf of a PyTree and return a PyTree of LinearAutoEncoder instances.

        Args:
            tree (PyTree): The input tree with arrays to fit the autoencoder on. Each array must be at most 2D. The first dimension of each array is assumed to be the sample dimension.
            n_latent (int | PyTree): The number of latent dimensions to reduce to for each leaf. If an int is provided, the same number of latent dimensions is used for all leaves. If a PyTree is provided, it must have the same structure as `tree`.
            scaling_type (ScalingType | str, optional): The type of scaling to apply to the data before fitting. Defaults to ScalingType.STD.
        """

        def fit_fn(x, n_latent):
            return LinearAutoEncoder.fit(x, n_latent, scaling_type)[0]

        if isinstance(n_latent, int):
            n_latent_tree = jax.tree.map(lambda _: n_latent, tree)
        else:
            n_latent_tree = n_latent

        encoders = jax.tree.map(fit_fn, tree, n_latent_tree, is_leaf=lambda x: isinstance(x, Array))
        return encoders
