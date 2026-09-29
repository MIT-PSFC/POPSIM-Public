import math

import jax
import loguru
import xarray as xr
from jaxtyping import Array

from popsim import TimeIndepModule
from popsim.ml.dataloading import DataLoader
from popsim.ml.flattener import VariableFlattener
from popsim.ml.rtd_activation import Activation
from popsim.ml.rtd_mlp import RtdMLP
from popsim.modules.linear_autoencoder import choose_n_latent
from popsim.norm_data import ScalingType

MIN_WIDTH = 16
MAX_WIDTH = 512


def log_latent_choice(n_latent: int, cumulative_ratio, n_flat: int):
    """Log the chosen latent size and where the covariance spectrum reaches common variance targets."""
    knees = {target: int(min(len(cumulative_ratio), (cumulative_ratio < target).sum() + 1)) for target in (0.95, 0.99, 0.999)}
    loguru.logger.info(
        f"Latent size {n_latent} for {n_flat} flattened features. "
        f"Components needed for 95% / 99% / 99.9% variance: {knees[0.95]} / {knees[0.99]} / {knees[0.999]}."
    )


class MLPAutoEncoder(TimeIndepModule):
    """An MLP autoencoder over the normalized, flattened variables of one sample."""

    flattener: VariableFlattener
    encoder: RtdMLP
    decoder: RtdMLP

    def reconstruct_flat(self, x_flat: Array) -> Array:
        return self.decoder(self.encoder(x_flat))

    def squared_normalized_error(self, inputs: xr.Dataset | dict[str, Array]) -> Array:
        """Elementwise squared reconstruction error in normalized units, shape (n_flat,)."""
        x_flat = self.flattener.normalize_flat(inputs)
        x_recon = self.reconstruct_flat(x_flat)
        return (x_recon - x_flat) ** 2

    def __call__(self, inputs: xr.Dataset | dict[str, Array]) -> dict[str, xr.Variable]:
        """Reconstruction of every variable in physical units."""
        x_flat = self.flattener.normalize_flat(inputs)
        recon = self.flattener.unflatten_unnorm(self.reconstruct_flat(x_flat))
        return {name: xr.Variable(dims=self.flattener.dims_of(name), data=recon[name]) for name in self.flattener.variables}

    def trainable(self) -> tuple[RtdMLP, RtdMLP]:
        """The parts to train, normalization statistics stay frozen."""
        return (self.encoder, self.decoder)

    @classmethod
    def init(
        cls,
        train_dl: DataLoader,
        latent_size: int | None = None,
        explained_variance: float = 0.99,
        width_size: int | None = None,
        depth: int = 2,
        activation: Activation | str = Activation.RELU,
        scaling_type: ScalingType | str = ScalingType.QUANTILE_50,
        n_covariance_samples: int = 10_000,
        prng_seed: int = 0,
    ) -> "MLPAutoEncoder":
        """Build an autoencoder sized from the training data.

        The latent size is the number of principal components explaining `explained_variance` of the normalized
        training data unless given, and the hidden width is the geometric mean of the input and latent sizes unless given.
        """
        flattener = VariableFlattener.from_dataloader(train_dl, train_dl.metadata.input_vars, scaling_type)
        n_flat = flattener.n_flat

        if latent_size is None:
            rows = flattener.normalized_rows(train_dl.ds, n_covariance_samples, seed=prng_seed)
            latent_size, cumulative_ratio = choose_n_latent(rows, explained_variance)
            log_latent_choice(latent_size, cumulative_ratio, n_flat)
        if width_size is None:
            width_size = int(min(max(round(math.sqrt(n_flat * latent_size)), MIN_WIDTH), MAX_WIDTH))
        loguru.logger.info(f"MLPAutoEncoder: {n_flat} -> {latent_size} -> {n_flat}, width {width_size}, depth {depth}.")

        key_encoder, key_decoder = jax.random.split(jax.random.PRNGKey(prng_seed))
        activation = Activation(activation)
        encoder = RtdMLP(n_flat, latent_size, width_size, depth, activation=activation, key=key_encoder)
        decoder = RtdMLP(latent_size, n_flat, width_size, depth, activation=activation, key=key_decoder)
        return cls(flattener=flattener, encoder=encoder, decoder=decoder)
