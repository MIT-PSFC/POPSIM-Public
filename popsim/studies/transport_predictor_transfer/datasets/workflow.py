import os
import shutil
from abc import abstractmethod

import numpy as np
import xarray as xr
from loguru import logger

from popsim.data.dataset_utils import build_tensorized_dataset
from popsim.ml.preprocess_utils import mask_to_largest_group_mask
from popsim.ml.split_utils import split_dataset_by_fracs

DATASET_DT = 1e-3  # 1ms uniform timebase
TIME_DIM = "time_idx"
TIME_COORD = "time"
EPISODE_DIM = "shot"
MIN_SHOT_DURATION = 0.5  # seconds, minimum shot duration to include in dataset
TEST_SPLIT = (0.8, 0.2)  # Amount of data to reserve for testing on target device
TRAIN_VAL_SPLIT = (0.8, 0.2)  # Amount of training data to use for training vs validation
DATA_TYPE = np.float32  # All the TCV and CMOD data is in float32 anyway, do the whole study in float32


class DataWorkflow:
    """Class to handle organization of data processing steps.

    For this study, the general workflow is:
    1. Go from raw data to a tensorized dataset
    2. Process signals (renaming, unit conversions, filtering, etc.)
    3. Separate out test set on target device (completely held out from the rest)
    4. Place processed sets on scratch for fast access during training
    """

    def __init__(
        self,
        ds_name: str,
        is_target: bool,
        max_num_shots: int,
        min_shot_id: int | None,
        raw_ds_dir: str | None,
        processed_ds_dir: str | None,
        scratch_dir: str | None,
        extrapolate: str = "chronological",
    ):
        """
        Args:
            ds_name (str): Name of the dataset (e.g., "tcv_test", "cmod_200").
            is_target (bool): Whether this workflow is for the target device (True) or source device (False).
            max_num_shots (int): Maximum number of shots to include in the dataset.
            min_shot_id (int): Minimum shot ID to include in the dataset.
            raw_ds_dir (str): Directory to save the tensorized dataset.
            processed_ds_dir (str): Directory to save the processed dataset.
            scratch_dir (str): Directory to place the final datasets for training.
            extrapolate (str): Extrapolation for the test set, either "chronological" or "performance".
        """
        self.ds_name = ds_name
        self.is_target = is_target
        self.max_num_shots = max_num_shots
        self.min_shot_id = min_shot_id

        self.raw_ds_dir = raw_ds_dir
        self.raw_ds_path = os.path.join(raw_ds_dir, f"{ds_name}_raw.zarr")

        if processed_ds_dir:
            self.processed_ds_dir = processed_ds_dir
            self.processed_ds_path = os.path.join(processed_ds_dir, f"{ds_name}_processed.zarr")
        else:
            logger.warning(f"No processed_ds_dir provided for workflow {ds_name}")

        if scratch_dir:
            self.scratch_dir = scratch_dir
            if self.is_target:
                self.train_ds_path = os.path.join(scratch_dir, f"{ds_name}_train.zarr")
                self.test_ds_path = os.path.join(scratch_dir, f"{ds_name}_test.zarr")
                self.extrapolate = extrapolate
            else:
                self.train_ds_path = os.path.join(scratch_dir, f"{ds_name}_source.zarr")

        else:
            logger.warning(f"No scratch_dir provided for workflow {ds_name}")

    @abstractmethod
    def build_dataset(self, extend_existing: bool = False):
        """Build the tensorized dataset from raw data."""

    @abstractmethod
    def process_signals(self, ds: xr.Dataset) -> xr.Dataset:
        """Process signals from the raw dataset to the values expected by the module."""

    def process_dataset(self):
        """Process the dataset (e.g., filtering, normalization)."""
        logger.info(f"Processing dataset {self.ds_name} from {self.raw_ds_dir}")

        raw_ds = xr.open_zarr(self.raw_ds_path)

        def _process_fn(shot_id: int) -> xr.Dataset:
            shot_ds = raw_ds.sel({EPISODE_DIM: shot_id})

            # Convert the shot ds to non-dask arrays to avoid issues with some operations
            shot_ds = shot_ds.compute()
            # Process signals according to the provided signal map
            shot_ds = self.process_signals(shot_ds)
            # Reject unrealistic and/or unreliable datapoints
            shot_ds = apply_bounds_validation(shot_ds, self.signal_bounds)

            # Remove shots that are too short
            time = shot_ds[TIME_COORD]
            if (time.size == 0) or (time.max() - time.min() < MIN_SHOT_DURATION):
                return None

            return shot_ds

        processed_ds = build_tensorized_dataset(
            process_fn=_process_fn,
            identifiers=raw_ds["shot"].data,
            zarr_path=self.processed_ds_path,
            time_dim=TIME_DIM,
            episode_dim=EPISODE_DIM,
            extend_existing=False,
            mb_per_chunk=None,
        )

        logger.info(f"Completed processing dataset\n{processed_ds}")

    def separate_test_set(self):
        """Separate out the test set on the target device and places both on scratch."""

        ds = xr.open_zarr(self.processed_ds_path, consolidated=True)
        logger.info(f"Opened processed dataset with {ds.sizes['shot']} shots.")

        if self.extrapolate not in ["chronological", "performance"]:
            raise ValueError(f"Unknown extrapolation method: {self.extrapolate}")
        elif self.extrapolate == "chronological":
            logger.info("Separating test set chronologically by shot number.")
            sortby = "shot"
        elif self.extrapolate == "performance":
            logger.info("Separating test set by performance (Wtot**2 + Ip**2).")
            ds["performance"] = ds["Wtot_MJ"] ** 2 + ds["Ip_MA"] ** 2
            sortby = "performance"

        train_ds, test_ds = split_dataset_by_fracs(
            ds,
            fracs=TEST_SPLIT,
            dim="shot",
            seed=42,
            sortby=sortby,
        )

        logger.info(f"Train set has {train_ds.sizes['shot']} shots, test set has {test_ds.sizes['shot']} shots.")
        train_ds.to_zarr(self.train_ds_path, mode="w", consolidated=True)
        test_ds.to_zarr(self.test_ds_path, mode="w", consolidated=True)
        logger.info(f"Placed train set at {self.train_ds_path}")
        logger.info(f"Placed test set at {self.test_ds_path}")

    def place_on_scratch(self, ds_start_path, ds_end_path):
        """Place the datasets on scratch for fast access during training."""
        if os.path.exists(ds_end_path):
            logger.info(f"Dataset already exists at {ds_end_path}, skipping copy.")
        else:
            logger.info(f"Copying dataset to {ds_end_path}")
            ds = xr.open_zarr(ds_start_path, consolidated=True)
            ds.to_zarr(ds_end_path, mode="w", consolidated=True)

    def cleanup(self, cleanup: str):
        """Clean up intermediate datasets to save space.

        Args:
            cleanup (str): Stages to start clean up before running. Options are "raw", "processed", "scratch".
        """

        if cleanup == "raw" and hasattr(self, "raw_ds_dir"):
            cleanup = "processed"
            shutil.rmtree(self.raw_ds_path, ignore_errors=True)
            raw_ds_log = os.path.join(self.raw_ds_dir, f"build_{self.ds_name}.log")
            if os.path.exists(raw_ds_log):
                os.remove(raw_ds_log)
        if cleanup == "processed" and hasattr(self, "processed_ds_dir"):
            cleanup = "scratch"
            shutil.rmtree(self.processed_ds_path, ignore_errors=True)
        if cleanup == "scratch" and hasattr(self, "scratch_dir"):
            shutil.rmtree(self.train_ds_path, ignore_errors=True)
            if self.is_target:
                shutil.rmtree(self.test_ds_path, ignore_errors=True)

    def run(self):
        """Run the full data workflow, putting all datasets on the present cluster."""
        logger.info(f"Running workflow for dataset {self.ds_name}")

        if os.path.exists(self.raw_ds_dir):
            if not os.path.exists(self.raw_ds_path):
                logger.info(f"Building raw dataset in {self.raw_ds_dir}")
                self.build_dataset()
        else:
            logger.warning(f"Raw dataset directory {self.raw_ds_dir} does not exist, skipping building raw dataset.")

        if os.path.exists(self.processed_ds_dir):
            if not os.path.exists(self.processed_ds_path):
                logger.info(f"Processing dataset in {self.processed_ds_dir}")
                self.process_dataset()
            else:
                logger.info(f"Processed dataset already exists at {self.processed_ds_path}, skipping processing dataset.")
        else:
            logger.warning(f"Processed dataset directory {self.processed_ds_dir} does not exist, skipping processing dataset.")

        if os.path.exists(self.scratch_dir):
            if self.is_target and (not os.path.exists(self.train_ds_path) or not os.path.exists(self.test_ds_path)):
                logger.info("Target device, separating test set.")
                self.separate_test_set()
            elif not os.path.exists(self.train_ds_path):
                logger.info("Source device, no test set separation needed.")
                self.place_on_scratch(self.processed_ds_path, self.train_ds_path)
        else:
            logger.warning(f"Scratch directory {self.scratch_dir} does not exist, skipping placing datasets on scratch.")

    def run_data_cluster(self):
        """Run the data workflow from raw -> processed, assuming it is on the data cluster."""

        if os.path.exists(self.processed_ds_path):
            logger.info(f"Processed dataset already exists at {self.processed_ds_path}, nothing to do on data cluster.")
            return

        # Check if the raw dataset already exists
        if os.path.exists(self.raw_ds_path):
            logger.info(f"Raw dataset already exists at {self.raw_ds_path}, skipping building raw dataset.")
        else:
            logger.info(f"Building raw dataset in {self.raw_ds_dir}")
            self.build_dataset()

        logger.info(f"Processing dataset in {self.processed_ds_dir}")
        self.process_dataset()

    def run_compute_cluster(self):
        """Run the data workflow from processed -> scratch, assuming it is on the compute cluster."""

        if not os.path.exists(self.processed_ds_path):
            raise ValueError(f"Processed dataset does not exist at {self.processed_ds_path}, cannot proceed on compute cluster.")

        if self.is_target:
            if os.path.exists(self.train_ds_path) and os.path.exists(self.test_ds_path):
                logger.info(f"Train and test datasets already exist in {self.scratch_dir}, nothing to do on compute cluster.")
                return
            logger.info("Target device, separating test set.")
            self.separate_test_set()
        else:
            if os.path.exists(self.train_ds_path):
                logger.info(f"Train dataset already exists in {self.scratch_dir}, nothing to do on compute cluster.")
                return
            logger.info("Source device, no test set separation needed.")
            self.place_on_scratch(self.processed_ds_path, self.train_ds_path)


def apply_bounds_validation(shot_ds: xr.Dataset, signal_bounds: dict[str, dict]) -> xr.Dataset:
    for signal_name, config in signal_bounds.items():
        data = shot_ds[signal_name]
        min_val, max_val = config["bounds"]

        # Select specific dimension values if specified
        if "select" in config:
            for dim, value in config["select"].items():
                data = data.sel({dim: value})

        valid_condition = xr.ones_like(data, dtype=bool)
        if min_val is not None:
            valid_condition = valid_condition & (data > min_val)
        if max_val is not None:
            valid_condition = valid_condition & (data < max_val)

        shot_ds[f"{signal_name}_valid"] = valid_condition

    valid_vars = [shot_ds[f"{name}_valid"] for name in signal_bounds.keys()]
    shot_ds["valid_data"] = xr.concat(valid_vars, dim="temp").all(dim="temp")
    shot_ds["valid_data_largest_group"] = mask_to_largest_group_mask(shot_ds["valid_data"], episode_dim="shot", time_dim="time_idx")
    shot_ds = shot_ds.where(shot_ds["valid_data_largest_group"], drop=True)
    shot_ds = shot_ds.drop_vars([f"{name}_valid" for name in signal_bounds.keys()] + ["valid_data", "valid_data_largest_group"])

    return shot_ds
