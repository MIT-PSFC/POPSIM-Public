"""
Code to interface into equil data from MEQ provided by Dan Boyer.
Canonical example is in popsim/data/equil_data_101.mat
"""
import collections
import os
import typing

import chex
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
from jaxtyping import Array, ArrayLike, PyTree

from popsim import PACKAGE_ROOT
from popsim.gui import PopsimGUI
from popsim.interp import interp_trees
from popsim.tree_util import build_ordered_dict
from popsim.xarray_utils import time_and_pytree_to_xarray


def to_single_array(arr: Array) -> Array:
    """By default, the loaded data is an array of arrays.
    Transform it into a single array where the time dimension is the first dimension.

    Args:
        arr (Array): _description_

    Returns:
        Array: _description_
    """
    time_slices = tuple(np.squeeze(arr))
    return np.squeeze(np.stack(time_slices))


def labels_to_list(labels: Array) -> list:
    """String labels from the .mat file are in an annoying format. Reformat."""
    return [str(np.squeeze(x)) for x in np.squeeze(labels)]


def replace_in_list(list_, val, replace):
    return [replace if x == val else x for x in list_]


ArrayOrPath = typing.Union[Array, diffrax.AbstractPath]


@chex.dataclass
class Params:
    plasma_resistance: ArrayLike
    non_inductive_current: ArrayLike


@chex.dataclass
class Control:
    labels: Array
    values: Array


@chex.dataclass
class State:
    coil_labels: Array
    vessel_labels: Array
    values: Array

    @property
    def plasma_current(self) -> ArrayLike:
        # Plasma current is the last value in the state by convention.
        return self.values[-1]


@chex.dataclass
class LTVEquil:
    """
    Dataclass for the Linear Time-Varying Equilibrium data.
    See math in: https://github.com/cfs-energy-internal/POPSIM/issues/25
    """

    time: ArrayOrPath  # Times for which the data is valid.
    Amod: ArrayOrPath  # Dynamics matrix for non-current components.
    Ares: ArrayOrPath  # Dynamics matrix for current component.
    Bmod: ArrayOrPath  # Input matrix for non-current components.
    Bres: ArrayOrPath  # Input matrix for inductive current component.
    C: ArrayOrPath  # "C" output matrix for current coils, etc.
    D: ArrayOrPath  # "D" output matrix for current coils, etc.
    C_Fx: ArrayOrPath  # "C" output matrix for flux grid
    D_Fx: ArrayOrPath  # "D" output matrix for flux grid
    y0: ArrayOrPath  # Feed-forward output for current coils, etc.
    y0_Fx: ArrayOrPath  # Feed-forward output for flux grid
    betap: ArrayOrPath  # Feed-forward beta_p.
    li: ArrayOrPath  # Feed-forward li.
    li_dot: ArrayOrPath  # Derivative of li.
    betap_dot: ArrayOrPath  # Derivative of bp.
    x0: ArrayOrPath  # Feed-forward trajectory.

    def build_interpolation(self, method="linear") -> diffrax.AbstractPath:
        return interp_trees(self.time, self, method)


def dynamics(equil_slice: LTVEquil, state: Array, control: Array, params: Params) -> Array:
    state_dot = (
        equil_slice.Amod @ state
        + params.plasma_resistance * equil_slice.Ares @ state.plasma_current
        + equil_slice.Bmod @ control
        + params.plasma_resistance * equil_slice.Bres @ params.non_inductive_current
    )
    return state_dot


@chex.dataclass(frozen=True)
class TreeBuilder:
    coil_current_labels: list[str]
    vessel_mode_labels: list[str]
    Bm_labels: list[str]
    Ff_labels: list[str]
    control_labels: list[str]

    def build_state_tree(self, arr: Array) -> PyTree[ArrayLike]:
        coil_current = arr[: len(self.coil_current_labels)]
        vessel_modes = arr[len(self.coil_current_labels) : len(self.coil_current_labels) + len(self.vessel_mode_labels)]
        IP = arr[-1]
        out = {
            "coil_currents": build_ordered_dict(self.coil_current_labels, coil_current),
            "vessel_modes": build_ordered_dict(self.vessel_mode_labels, vessel_modes),
            "IP": jnp.array(IP),
        }
        return collections.OrderedDict(out)

    def build_output_tree(self, arr: Array) -> PyTree[ArrayLike]:
        Ia = arr[: len(self.coil_current_labels)]
        Bm_end = len(self.coil_current_labels) + len(self.Bm_labels)
        Bm = arr[len(self.coil_current_labels) : Bm_end]
        Ff = arr[Bm_end : Bm_end + len(self.Ff_labels)]
        out = {
            "coil_currents": build_ordered_dict(self.coil_current_labels, Ia),
            "Bm": build_ordered_dict(self.Bm_labels, Bm),
            "Ff": build_ordered_dict(self.Ff_labels, Ff),
            "Ip": arr[-4],
            "Ft": arr[-3],
            "rIp": arr[-2],
            "zIp": arr[-1],
        }
        return collections.OrderedDict(out)

    def build_control_tree(self, arr: Array) -> PyTree[ArrayLike]:
        return build_ordered_dict(self.control_labels, arr)


def load():
    data = sio.loadmat(os.path.join(PACKAGE_ROOT, "data", "equil_data_101.mat"))
    metadata = sio.loadmat(os.path.join(PACKAGE_ROOT, "data", "equil_data_101_meta.mat"))
    # Rename keys.
    data["y0"] = data.pop("yo")
    data["y0_Fx"] = data.pop("yo_Fx")
    data["betap"] = data.pop("bp")

    trajectories = {
        k: v for k, v in data.items() if k in ["Amod", "Ares", "Bmod", "Bres", "C", "D", "C_Fx", "D_Fx", "y0", "y0_Fx", "betap", "li", "x0"]
    }
    time = np.squeeze(data["time"])
    # What the [2:]? Some of the arrays have extra time steps relative to time, with the first two having a different shape for some reason.
    trajectories = {k: np.squeeze(v)[2:] if v.size == (time.size + 2) else v for k, v in trajectories.items()}

    # Matlab creates these nested array things. Convert to single arrays.
    trajectories = jax.tree_map(lambda arr: to_single_array(arr), trajectories)
    # Perform an interpolation to get the derivatives.
    li_interp = interp_trees(time, trajectories["li"], "cubic")
    bp_interp = interp_trees(time, trajectories["betap"], "cubic")
    trajectories["li_dot"] = li_interp.derivative(time)
    trajectories["betap_dot"] = bp_interp.derivative(time)

    equil = LTVEquil(time=time, **trajectories)
    equil = interp_trees(equil.time, equil, "linear")

    current_labels = labels_to_list(metadata["dima"])
    vessel_mode_labels = labels_to_list(metadata["dimu"])
    control_labels = labels_to_list(metadata["InputNames"])

    # Rename some of the control labels.
    control_labels = replace_in_list(control_labels, "Co_bp_001", "betap")
    control_labels = replace_in_list(control_labels, "Co_li_001", "li")
    control_labels = replace_in_list(control_labels, "Ini_001_S", "Ini")
    control_labels = replace_in_list(control_labels, "dCodt_bp_001", "betap_dot")
    control_labels = replace_in_list(control_labels, "dCodt_li_001", "li_dot")

    Bm_labels = labels_to_list(metadata["dimm"])
    Ff_labels = labels_to_list(metadata["dimf"])
    tree_builder = TreeBuilder(
        coil_current_labels=current_labels,
        vessel_mode_labels=vessel_mode_labels,
        Bm_labels=Bm_labels,
        Ff_labels=Ff_labels,
        control_labels=control_labels,
    )
    return equil, tree_builder


if __name__ == "__main__":
    import panel as pn

    equil, tree_builder = load()
    ts = jnp.linspace(equil.ts[0], equil.ts[-1], 100)
    equil_slices = jax.vmap(equil.evaluate)(ts)

    states = jax.vmap(tree_builder.build_state_tree)(equil_slices.x0)
    outputs = jax.vmap(tree_builder.build_output_tree)(equil_slices.y0)
    betap = equil_slices.betap
    li = equil_slices.li

    tree = {
        "states": states,
        "outputs": outputs,
        "betap": betap,
        "li": li,
    }

    ds = time_and_pytree_to_xarray(ts, tree, multi_simulation=False)
    gui_handle = PopsimGUI(ds, time_dim="time")
    pn.serve(gui_handle.build_view(), port=8080, show=False)
