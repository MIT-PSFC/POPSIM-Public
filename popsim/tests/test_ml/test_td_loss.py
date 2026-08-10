"""
Test that time-dependent loss functions properly handle variable-length shots with NaN padding.

This test creates a synthetic dataset with shots of different lengths and verifies
that the loss function properly ignores forward-filled NaN padding at the end of shots.
"""
import time

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import xarray as xr

from popsim.ml.dataloading import ffill_end_of_time_padding, make_time_dep_dataloader
from popsim.ml.eval import batched_model_eval_and_loss
from popsim.ml.loss import IntegralLoss
from popsim.ml.preprocess_utils import force_drop_nans
from popsim.modules.power_balance.module import PowerBalance, PowerBalanceEnv
from popsim.modules.power_balance.p_oh.module import OhmicPower
from popsim.modules.power_balance.p_rad.module import RadiatedPower
from popsim.modules.profile_predictor.module import Outputs as ProfilePredictorOutputs
from popsim.modules.transport_predictor.module import TransportPredictor, TransportPredictorEnv
from popsim.simulate import StepperType
from popsim.tests.fixtures import load_cmod_test_dataset
from popsim.tree_util import any_nans
from popsim.utils import time_epsilon


def create_variable_length_dataset(num_shots: int = 4, shadowed_inputs: bool | None = False, shadowed_0D_targets: bool | None = False, shadowed_1D_targets: bool | None = False, no_nan_time: bool | None = False, target_sentinel: float | None = None) -> xr.Dataset:
    """Synthetic dataset with ragged data along the time_idx dimension.
    This is created with all the data needed for power balance and transport predictor modules.
    
    Args:
        num_shots (int): Number of shots in the dataset.
        shadowed_inputs (bool): Whether to have some of the input features have more NaNs than others
        shadowed_0D_targets (bool): Whether to have the 0D targets have more NaNs than the inputs
        shadowed_1D_targets (bool): Whether to have the 1D targets have more NaNs than the inputs
        no_nan_time (bool): If the dataset is made without the tensorized dataset helper, the time coordinate might be non-NaN when everything else is.
        target_sentinel (float): Value to insert for the final valid timestep of the targets, to verify that forward-filled values are not being included.
    
    Returns:
        xr.Dataset: The synthetic dataset with variable-length shots and the specified NaN patterns.
    """

    shot_ids = np.arange(num_shots)

    # Calculate max time length for padding
    max_time_length = 6 + (num_shots - 1)

    shot_datasets = []
    for shot_id in shot_ids:
        time_length = 6 + shot_id  # Variable length for each shot
        time_idx = np.arange(time_length)

        time = time_idx / 10 # Tenths of seconds, to be distinct from time_idx

        # Power balance module
        input_vars_power_balance = {
            "Ip_MA": np.ones((time_length,)),
            "B0": np.ones((time_length,)),
            "ne19": np.ones((time_length,)),
            "P_abs_MW": np.ones((time_length,)),
            "R0": np.ones((time_length,)),
            "kappa": np.ones((time_length,)),
            "epsilon": np.ones((time_length,)),
            "delta_top": np.ones((time_length,)),
            "delta_bottom": np.ones((time_length,)),
            "P_rad_MW": np.ones((time_length,)),
        }

        # Additional signals for transport predictor module
        rhogrid = np.linspace(0, 1, 10)  # 1D spatial grid
        input_vars_transport_predictor = {
            "P_aux_MW": np.ones((time_length,)),
            "ne20": np.ones((time_length,)),
            "a_minor": np.ones((time_length,)),
        }

        # Target variables for transport predictor (the power balance is included here too)
        target_vars = {
            "Te_keV_rho": np.ones((time_length, len(rhogrid))),
            "ne20_rho": np.ones((time_length, len(rhogrid))),
            "P_oh_MW": np.ones((time_length,)),
            "Wtot_MJ": np.ones((time_length,)),
            "P_rad_MW": np.ones((time_length,)),
        }

        if shadowed_inputs:
            # Introduce NaNs in some input features
            for var in ["R0", "kappa"]:
                input_vars_power_balance[var][-2:] = np.nan  # Last two time steps are NaN

        if shadowed_0D_targets:
            # Introduce NaNs in 0D targets
            target_vars["Wtot_MJ"][-3:] = np.nan  # Last three time steps are NaN

        if shadowed_1D_targets:
            # Introduce NaNs in 1D targets
            target_vars["Te_keV_rho"][-4:, 5:] = np.nan  # Last four time steps and half of the spatial grid are NaN
            target_vars["ne20_rho"][-4:, :] = np.nan # Last four time steps are all NaN

        if target_sentinel is not None:
            for var in ["Te_keV_rho", "ne20_rho", "Wtot_MJ"]:
                last_valid_idx = np.where(~np.isnan(target_vars[var]))[0][-1]
                if var == "Wtot_MJ":
                    target_vars[var][last_valid_idx] = target_sentinel
                else:
                    target_vars[var][last_valid_idx, :] = target_sentinel
                # Also set the 'time' for this index to be extremely close to the previous one,
                # This is so traprule integral loss is small, unless it's being improperly forward-filled
                time[last_valid_idx] = np.nextafter(time[last_valid_idx - 1], np.inf)

        # Combine all variables into a single dataset for the shot
        # Note: P_rad_MW is both an input for power_balance and a target for transport_predictor
        # We include it as a target here
        shot_ds = xr.Dataset(
            data_vars={
                **{name: (("shot", "time_idx"), data[np.newaxis, :]) for name, data in input_vars_power_balance.items() if name != "P_rad_MW"},
                **{name: (("shot", "time_idx"), data[np.newaxis, :]) for name, data in input_vars_transport_predictor.items()},
                "Wtot_MJ": (("shot", "time_idx"), target_vars["Wtot_MJ"][np.newaxis, :]),
                "P_oh_MW": (("shot", "time_idx"), target_vars["P_oh_MW"][np.newaxis, :]),
                "P_rad_MW": (("shot", "time_idx"), target_vars["P_rad_MW"][np.newaxis, :]),
                "Te_keV_rho": (("shot", "time_idx", "rho"), target_vars["Te_keV_rho"][np.newaxis, :, :]),
                "ne20_rho": (("shot", "time_idx", "rho"), target_vars["ne20_rho"][np.newaxis, :, :]),
                "time": (("shot", "time_idx"), time[np.newaxis, :]),
            },
            coords={
                "shot": [shot_id],
                "rho": rhogrid,
            },
        )

        # Pad to max_time_length with NaN, like would happen when making a tensorized dataset
        if time_length < max_time_length:
            shot_ds = shot_ds.pad(time_idx=(0, max_time_length - time_length), mode='constant', constant_values=np.nan)

        shot_datasets.append(shot_ds)


    # Concatenate all shots into a single dataset
    full_ds = xr.concat(shot_datasets, dim="shot")
    
    full_ds["ne_shape"] = full_ds["ne20_rho"] / full_ds["ne20_rho"].integrate("rho")
    full_ds["Te_shape"] = full_ds["Te_keV_rho"] / full_ds["Te_keV_rho"].integrate("rho")

    if no_nan_time:
        # Overwrite all nan times with time_idx/10, to make a dataset where the time coordinate is never NaN, even though all the inputs and targets are
        # Create a broadcast-compatible replacement array
        time_idx_array = np.arange(full_ds.sizes["time_idx"]) / 10
        time_replacement = xr.DataArray(
            np.broadcast_to(time_idx_array[np.newaxis, :], full_ds["time"].shape),
            dims=full_ds["time"].dims,
            coords={"shot": full_ds.coords["shot"]}
        )
        full_ds["time"] = xr.where(~np.isnan(full_ds["time"]), full_ds["time"], time_replacement)
    
    full_ds = full_ds.set_coords("time") # Promote "time" to a coordinate, like we would do when loading from a zarr store

    return full_ds


class SimpleMockTransportPredictor(eqx.Module):
    """A simple mock transport predictor that returns constant values.
    This is used for testing the loss function and data pipeline without
    needing to load the full TransportPredictor model.
    """
    rhogrid: jnp.ndarray

    State = TransportPredictor.State
    Output = TransportPredictor.Output

    def __init__(self, rhogrid):
        self.rhogrid = jnp.array(rhogrid)

    def __call__(self, state, inputs):
        """Return mock outputs with all values set to 1.0"""
        # Create mock outputs with the expected structure
        profile_predictor_output = ProfilePredictorOutputs(
            ne=jnp.ones_like(self.rhogrid),
            te=jnp.ones_like(self.rhogrid),
            debug_info=None,
        )

        power_balance_output = PowerBalance.Output(
            Wtot_MJ_pred=1.0,
            P_cond_MW=1.0,
            taue_predictor_output=None,
        )

        p_oh_output = OhmicPower.Output(
            P_oh_MW_pred=1.0,
            debug_info={},
        )

        p_rad_output = RadiatedPower.Output(
            P_rad_MW_pred=1.0,
            debug_info={},
        )

        output = TransportPredictor.Output(
            profile_predictor_output=profile_predictor_output,
            power_balance_output=power_balance_output,
            p_oh_output=p_oh_output,
            p_rad_output=p_rad_output,
            rho=self.rhogrid,
        )

        power_balance_state_dot = PowerBalance.State(Wtot_MJ=0.0)
        state_dot = TransportPredictor.State(power_balance_state=power_balance_state_dot)

        return state_dot, output


def basic_dataloader_module_loss(module_name: str, ds: xr.Dataset):
    """Setup a basic dataloader, module, and loss function for these tests"""

    if module_name == "power_balance":
        ds = force_drop_nans(ds, data_vars=["Ip_MA", "B0", "ne19", "P_abs_MW", "R0", "kappa", "epsilon", "delta_top", "delta_bottom", "P_rad_MW", "Wtot_MJ"], time_var_dim="time_idx", episode_var_dim="shot", time_coord="time")
        dl = make_time_dep_dataloader(
            ds=ds,
            time_coord="time",
            episode_coord="shot",
            state_init_vars=["Wtot_MJ"],
            input_vars=["Ip_MA", "B0", "ne19", "P_abs_MW", "R0", "kappa", "epsilon", "delta_top", "delta_bottom", "P_rad_MW"],
            target_vars=["Wtot_MJ"],
            convert_xr_to_jnp=False,
            segment_length=5, # Segment length less than the shortest shot
            segment_overlap=1,
            batch_size=2,
            shuffle=False, # Don't shuffle so we can verify the contents of the batches
        )
        def loss_fn(pred, targ):
            absolute_error = jnp.abs(pred.Wtot_MJ_pred - targ["Wtot_MJ"].data)
            return optax.huber_loss(absolute_error)
        module = PowerBalance.init({
            "model_type": "scaling_law",
            "min_val": 0.001,
            "max_val": 0.12,
        })
        env = PowerBalanceEnv(module=module, stepper=StepperType.SIMPLE_EULER)
    elif module_name == "transport_predictor":
        ds = force_drop_nans(ds, data_vars=["Ip_MA", "B0", "ne19", "P_abs_MW", "R0", "kappa", "epsilon", "delta_top", "delta_bottom", "P_rad_MW", "P_aux_MW", "ne20", "a_minor", "Te_keV_rho", "ne20_rho", "P_oh_MW", "Wtot_MJ"], time_var_dim="time_idx", episode_var_dim="shot", time_coord="time")
        dl = make_time_dep_dataloader(
            ds=ds,
            time_coord="time",
            episode_coord="shot",
            state_init_vars=["Wtot_MJ"],
            input_vars=["R0", "B0", "Ip_MA", "a_minor", "kappa", "delta_top", "delta_bottom", "P_aux_MW", "ne20"],
            target_vars=["Te_keV_rho", "ne20_rho", "Wtot_MJ", "P_oh_MW", "P_rad_MW"],
            convert_xr_to_jnp=False,
            segment_length=5, # Segment length less than the shortest shot
            segment_overlap=1,
            batch_size=2,
            shuffle=False, # Don't shuffle so we can verify the contents of the batches
        )
        def loss_fn(pred, targ):
            ne_rho_loss = jnp.trapezoid(
                optax.huber_loss(pred.profile_predictor_output.ne, targ["ne20_rho"].data, delta=0.1),
                x=pred.rho,
            )
            te_rho_loss = jnp.trapezoid(
                optax.huber_loss(pred.profile_predictor_output.te, targ["Te_keV_rho"].data, delta=0.1),
                x=pred.rho,
            )
            stored_energy_error = jnp.abs(pred.power_balance_output.Wtot_MJ_pred - targ["Wtot_MJ"].data)
            stored_energy_loss = optax.huber_loss(stored_energy_error, delta=0.1)
            ohmic_error = jnp.abs(pred.p_oh_output.P_oh_MW_pred - targ["P_oh_MW"].data)
            p_oh_loss = optax.huber_loss(ohmic_error, delta=0.1)
            radiated_error = jnp.abs(pred.p_rad_output.P_rad_MW_pred - targ["P_rad_MW"].data)
            p_rad_loss = optax.huber_loss(radiated_error, delta=0.1)

            total_loss = ne_rho_loss + te_rho_loss + stored_energy_loss + p_oh_loss + p_rad_loss
            return total_loss

        rhogrid = ds.coords["rho"].values
        module = SimpleMockTransportPredictor(rhogrid=rhogrid)
        env = TransportPredictorEnv(module=module, stepper=StepperType.SIMPLE_EULER)

    else:
        raise ValueError(f"Unknown module name: {module_name}")
    
    return dl, env, loss_fn


def check_batches(dl, model, instantaneous_loss_fn):
    loss_fn = IntegralLoss(instantaneous_loss_fn)
    for i, batch in enumerate(dl):
        inputs, targets = batch.get_inputs_and_targets()

        assert not any_nans(inputs), f"NaNs found in inputs in batch {i}"
        assert not any_nans(targets), f"NaNs found in targets in batch {i}"

        batch_losses = batched_model_eval_and_loss(
            model=model, loss_fn=loss_fn, inputs=inputs, targets=targets
        )
        batch_loss = float(batch_losses.mean())

        assert not np.isnan(batch_loss), f"Batch loss is NaN for batch {i}"
        assert batch_loss < 100, f"Batch loss is unexpectedly high for batch {i}: {batch_loss}"

@pytest.mark.parametrize("module", ["power_balance", "transport_predictor"])
@pytest.mark.parametrize("target_sentinel", [None, 1e6])
def test_batching_no_shadowing(module, target_sentinel):
    """Test that the batching and loss calculation works when there is no raggedness within shots.
    All signals exist for all time within a shot. This is fine and will always work.
    
    Example:
    Shot 1:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, nan, nan, nan]
        input_1: [1, 1, 1, nan, nan, nan]
        input_2: [1, 1, 1, nan, nan, nan]
        target_0D: [1, 1, 1, nan, nan, nan]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
    Shot 2:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        input_1: [1, 1, 1, 1, 1, 1]
        input_2: [1, 1, 1, 1, 1, 1]
        target_0D: [1, 1, 1, 1, 1, 1]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    """
    ds = create_variable_length_dataset(num_shots=4, target_sentinel=target_sentinel)
    dl, env, loss_fn = basic_dataloader_module_loss(module, ds)
    check_batches(dl, env, loss_fn)

@pytest.mark.parametrize("module", ["power_balance", "transport_predictor"])
def test_batching_input_shadowing(module):
    """Test that the time-dependent dataloader can properly catch when there is raggedness within shots (some signals are shadowed by others)
    For time-indepdent dataloader the samples with a NaN in them are dropped, but for time-dependent dataloaders we don't want to
    drop an entire sample just because there's a couple of NaNs.
    
    Example:
    Shot 1:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, nan, nan, nan]
        input_1: [1, 1, 1, nan, nan, nan]
        input_2: [1, 1, 1, nan, nan, nan]
        target_0D: [1, 1, 1, nan, nan, nan]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
    Shot 2:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        input_1: [1, 1, 1, 1, 1, 1]
        input_2: [1, 1, 1, 1, nan, nan]  <- PROBLEM
        target_0D: [1, 1, 1, 1, 1, 1]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    
    """
    ds = create_variable_length_dataset(num_shots=4, shadowed_inputs=True)
    dl, env, loss_fn = basic_dataloader_module_loss(module, ds)
    check_batches(dl, env, loss_fn)

@pytest.mark.parametrize("module", ["power_balance", "transport_predictor"])
def test_batching_0D_target_shadowing(module):
    """Test that the batching and loss calculation works when the 0D targets have NaNs within shots
    
    Example:
    Shot 1:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, nan, nan, nan]
        input_1: [1, 1, 1, nan, nan, nan]
        input_2: [1, 1, 1, nan, nan, nan]
        target_0D: [1, 1, 1, nan, nan, nan]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
    Shot 2:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        input_1: [1, 1, 1, 1, 1, 1]
        input_2: [1, 1, 1, 1, 1, 1]
        target_0D: [1, 1, 1, nan, nan, nan]  <- PROBLEM
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    
    """
    ds = create_variable_length_dataset(num_shots=4, shadowed_0D_targets=True)
    dl, env, loss_fn = basic_dataloader_module_loss(module, ds)
    check_batches(dl, env, loss_fn)

@pytest.mark.parametrize("module", ["power_balance", "transport_predictor"])
def test_batching_1D_target_shadowing(module):
    """Test that the batching and loss calculation works when the 1D targets have NaNs within shots
    
    Example:
    Shot 1:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, nan, nan, nan]
        input_1: [1, 1, 1, nan, nan, nan]
        input_2: [1, 1, 1, nan, nan, nan]
        target_0D: [1, 1, 1, nan, nan, nan]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
    Shot 2:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        input_1: [1, 1, 1, 1, 1, 1]
        input_2: [1, 1, 1, 1, 1, 1]
        target_0D: [1, 1, 1, 1, 1, 1]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]  <- PROBLEM
    
    """
    ds = create_variable_length_dataset(num_shots=4, shadowed_inputs=False, shadowed_0D_targets=False, shadowed_1D_targets=True)
    dl, env, loss_fn = basic_dataloader_module_loss(module, ds)
    check_batches(dl, env, loss_fn)


@pytest.mark.parametrize("module", ["power_balance", "transport_predictor"])
@pytest.mark.parametrize("shadowed_inputs", [False, True])
@pytest.mark.parametrize("shadowed_0D_targets", [False, True])
@pytest.mark.parametrize("shadowed_1D_targets", [False, True])
def test_batching_no_nan_time(module, shadowed_inputs, shadowed_0D_targets, shadowed_1D_targets):
    """Test that the dataloader properly detects when the time coordinate is not NaN
    and doesn't accidentally include forward-filled values in the loss calculation.
    
    Example:
    Shot 1:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  <- PROBLEM
        input_1: [1, 1, 1, nan, nan, nan]
        input_2: [1, 1, 1, nan, nan, nan]
        target_0D: [1, 1, 1, nan, nan, nan]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]
    Shot 2:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        input_1: [1, 1, 1, 1, 1, 1]
        input_2: [1, 1, 1, 1, 1, 1]
        target_0D: [1, 1, 1, 1, 1, 1]
        target_1D: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    
    """
    ds = create_variable_length_dataset(
        num_shots=4,
        shadowed_inputs=shadowed_inputs,
        shadowed_0D_targets=shadowed_0D_targets,
        shadowed_1D_targets=shadowed_1D_targets,
        no_nan_time=True,
        target_sentinel=1e6
    )
    dl, env, loss_fn = basic_dataloader_module_loss(module, ds)
    check_batches(dl, env, loss_fn)

def test_batching_intermittent_nans():
    """Test that the dataloader properly drops samples with NaNs in the middle of shots, even if the time coordinate is not NaN.
    
    Example:
    Shot 1:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  
        input_1: [1.0, 1.1, 1.2, 1.1, 1.4, 1.5]
        input_2: [2.0, nan, 2.2, 2.3, 2.4, 2.5]  <- PROBLEM
        target_0D: [3.0, 3.1, nan, 3.3, 3.4, 3.5]  <- PROBLEM
        target_1D: [[4.00, 4.01, 4.02], [4.10, 4.11, 4.12], [4.20, 4.21, 4.22], [nan, nan, nan], [4.40, 4.41, 4.42], [4.50, 4.51, 4.52]] <- PROBLEM
    Shot 2:
        time_idx: [0, 1, 2, 3, 4, 5]
        time: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        input_1: [1.0, 1.1, 1.2, 1.1, 1.4, 1.5]
        input_2: [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        target_0D: [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
        target_1D: [[4.00, 4.01, 4.02], [4.10, 4.11, 4.12], [4.20, 4.21, 4.22], [4.30, 4.31, 4.32], [4.40, 4.41, 4.42], [4.50, 4.51, 4.52]]
    """

    ds_orig = xr.Dataset(
        data_vars={
            "input_1": (("shot", "time_idx"), np.array([[1.0, 1.1, 1.2, 1.1, 1.4, 1.5], [1.0, 1.1, 1.2, 1.1, 1.4, 1.5]])),
            "input_2": (("shot", "time_idx"), np.array([[2.0, np.nan, 2.2, 2.3, 2.4, 2.5], [2.0, 2.1, 2.2, 2.3, 2.4, 2.5]])),
            "target_0D": (("shot", "time_idx"), np.array([[3.0, 3.1, np.nan, 3.3, 3.4, 3.5], [3.0, 3.1, 3.2, 3.4, 3.4, 3.5]])),
            "target_1D": (("shot", "time_idx", "rho"), np.array([[[4.00, 4.01, 4.02], [4.10, 4.11, 4.12], [4.20, 4.21, 4.22], [np.nan, np.nan, np.nan], [4.40, 4.41, 4.42], [4.50, 4.51, 4.52]], [[4.00, 4.01, 4.02], [4.10, 4.11, 4.12], [4.20, 4.21, 4.22], [4.30, 4.31, 4.32], [4.40, 4.41, 4.42], [4.50, 4.51, 4.52]]])),
        },
        coords={
            "time": (("shot", "time_idx"), np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]])),
            "shot": [1, 2],
            "rho": [0, 0.5, 1],
        },
    )

    ds_dropped = force_drop_nans(ds_orig, data_vars=["input_1", "input_2", "target_0D", "target_1D"], time_var_dim="time_idx", episode_var_dim="shot", time_coord="time")

    dl = make_time_dep_dataloader(
        ds=ds_dropped,
        time_coord="time",
        episode_coord="shot",
        state_init_vars=[],
        input_vars=["input_1", "input_2"],
        target_vars=["target_0D", "target_1D"],
        convert_xr_to_jnp=False,
        segment_length=5,
        segment_overlap=1,
        batch_size=2,
        shuffle=False,
    )

    # Shot 1 should have been modified to mask to the largst contiguous non-NaN segment,
    # and have the remaining nans get forward-filled
    shot_1_expected = xr.Dataset(
        data_vars={
            "input_1": (("time_idx",), np.array([1.4, 1.5, 1.5, 1.5, 1.5, 1.5])),
            "input_2": (("time_idx",), np.array([2.4, 2.5, 2.5, 2.5, 2.5, 2.5])),
            "target_0D": (("time_idx",), np.array([3.4, 3.5, 3.5, 3.5, 3.5, 3.5])),
            "target_1D": (("time_idx", "rho"), np.array([[4.40, 4.41, 4.42], [4.50, 4.51, 4.52], [4.50, 4.51, 4.52], [4.50, 4.51, 4.52], [4.50, 4.51, 4.52], [4.50, 4.51, 4.52]])),
        },
        coords={
            "time": (("time_idx",), np.array([0.4, 0.5, 0.5, 0.5, 0.5, 0.5])), # Time should also be forward-filled, but with very small increments to avoid being exactly the same (which would mess with trap rule loss)
            "shot": [1],
            "rho": [0, 0.5, 1],
        },
    ).set_coords("time")
    shot_1_loaded = dl.ds.sel(shot=1)
    for var in ["input_1", "input_2", "target_0D", "target_1D", "time"]:
        assert np.allclose(shot_1_expected[var].values[:-1], shot_1_loaded[var].values[0], atol=1e-3), f"Variable {var} did not match expected values for shot 1"

    # Shot 2 should be unchanged
    shot_2_expected = ds_orig.sel(shot=2)
    shot_2_loaded = dl.ds.sel(shot=2)
    for var in ["input_1", "input_2", "target_0D", "target_1D", "time"]:
        assert np.allclose(shot_2_expected[var].values[:-1], shot_2_loaded[var].values[0], atol=1e-3), f"Variable {var} was unexpectedly modified for shot 2"


def test_ffill_eot_padding():
    ds_even = create_variable_length_dataset(num_shots=4)
    ds_padded = ffill_end_of_time_padding(ds_even, time_coord="time", time_dim="time_idx")
    assert not any_nans(ds_padded["time"].values), "Time coordinate still has NaNs after forward-filling"

    ds_ragged_0D = create_variable_length_dataset(num_shots=4, shadowed_inputs=True)
    ds_padded_0D = ffill_end_of_time_padding(ds_ragged_0D, time_coord="time", time_dim="time_idx")
    assert not any_nans(ds_padded_0D["time"].values), "Time coordinate still has NaNs after forward-filling"

    ds_ragged_1D = create_variable_length_dataset(num_shots=4, shadowed_1D_targets=True)
    ds_padded_1D = ffill_end_of_time_padding(ds_ragged_1D, time_coord="time", time_dim="time_idx")
    assert not any_nans(ds_padded_1D["time"].values), "Time coordinate still has NaNs after forward-filling"

def test_force_drop_cmod():
    ds = load_cmod_test_dataset()

    for drop_nans in [True, False]:
        start_time = time.time()

        if drop_nans:
            ds_dropped = force_drop_nans(ds, data_vars=["beta_n", "li"], time_var_dim="time_slice", episode_var_dim="shot", time_coord="time")
            assert xr.Dataset.equals(load_cmod_test_dataset(), ds) # Verify that the original dataset is unchanged by the dropping process
        else:
            ds_dropped = ds

        dl = make_time_dep_dataloader(
            ds=ds_dropped,
            time_coord="time",
            episode_coord="shot",
            state_init_vars=['beta_n'],
            input_vars=['beta_n', 'li'],
            target_vars=['li'],
            convert_xr_to_jnp=False,
            extra_vars=None,
            segment_length=None,
            segment_overlap=0,
            batch_size=None,
            shuffle=False,
            generate_prng=False,
        )

        drop_nans_time = time.time() - start_time
        print(f"Time taken with drop_nans={drop_nans}: {drop_nans_time:.2f} seconds")

        for batch in dl:
            inputs, targets = batch.get_inputs_and_targets()

            assert not any_nans(inputs), f"NaNs found in inputs with drop_nans={drop_nans}"
            assert not any_nans(targets), f"NaNs found in targets with drop_nans={drop_nans}"
