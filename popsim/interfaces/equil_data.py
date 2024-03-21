"""
Code to interface into equil data from MEQ provided by Dan Boyer.
Canonical example is in popsim/data/equil_data_101.mat
"""
import os

import equinox as eqx
import jax
import numpy as np
import scipy.io as sio
from jaxtyping import Array

from popsim import PACKAGE_ROOT
from popsim.tree_util import leaves_as_array


class LTVEquil(eqx.Module):
    Amod: Array
    Ares: Array
    Bmod: Array
    Bres: Array
    C_Fx: Array
    D_Fx: Array
    yo_Fx: Array

    def __call__(self, state, control, time):
        pass

    def output(self, state, control, time):
        return self.CFx(time) @ leaves_as_array(state) + self.DFx(time) @ leaves_as_array(control) + self.yoFx(time)


def to_single_array(arr: Array) -> Array:
    """By default, the loaded data is an array of arrays.
    Transform it into a single array where the time dimension is the first dimension.

    Args:
        arr (Array): _description_

    Returns:
        Array: _description_
    """
    return np.stack(*arr)


def load():
    data = sio.loadmat(os.path.join(PACKAGE_ROOT, "data", "equil_data_101.mat"))
    _ = sio.loadmat(os.path.join(PACKAGE_ROOT, "data", "equil_data_101_meta.mat"))
    matrices = {k: v for k, v in data.items() if k in ["Amod", "Ares", "Bmod", "Bres", "C_Fx", "D_Fx", "yo_Fx"]}

    matrices = jax.tree_map(lambda arr: to_single_array(arr)[1:-2], matrices)


if __name__ == "__main__":
    load()
