import chex
from numpy import float64
from numpy.typing import NDArray

from popsim.cfspopcon_jax.helpers import integrate_profile_over_volume


@chex.dataclass
class VolumeIntegrator:
    rho: NDArray[float64]
    dV_drho: NDArray[float64]

    def __init__(self, rho, dV_drho):
        self.rho = rho
        self.dV_drho = dV_drho

    def __call__(self, array_per_m3):
        return integrate_profile_over_volume(array_per_m3, self.rho, self.dV_drho)
