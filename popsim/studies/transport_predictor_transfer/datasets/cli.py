#!/usr/bin/env python3
"""CLI for creating datasets using the DataWorkflow class on either TCV or C-Mod with Fire."""

import fire
from loguru import logger

from popsim.data import get_path_to_ml_data_dump, get_path_to_ml_data_scratch
from popsim.studies.transport_predictor_transfer.datasets.cmod_data import CMODDataWorkflow
from popsim.studies.transport_predictor_transfer.datasets.tcv_data import TCVDataWorkflow

RAW_DS_DIR_DEFAULT = get_path_to_ml_data_dump()
PROCESSED_DS_DIR_DEFAULT = get_path_to_ml_data_dump()
SCRATCH_DIR_DEFAULT = get_path_to_ml_data_scratch()

MAX_NUM_SHOTS_DEFAULT = 8


class DatasetCLI:
    """CLI for creating datasets using DataWorkflow."""

    def tcv(
        self,
        ds_name: str,
        is_target: bool = True,
        max_num_shots: int = MAX_NUM_SHOTS_DEFAULT,
        min_shot_id: int = 53134,
        raw_ds_dir: str = RAW_DS_DIR_DEFAULT,
        processed_ds_dir: str = PROCESSED_DS_DIR_DEFAULT,
        scratch_dir: str = SCRATCH_DIR_DEFAULT,
        extrapolate: str = "chronological",
        cleanup: str | None = None,
        data_cluster_only: bool = False,
        compute_cluster_only: bool = False,
    ):
        """Create a TCV dataset using TCVDataWorkflow.

        Args:
            ds_name: Name of the dataset (e.g., "tcv_test", "tcv_1000").
            is_target: Whether this workflow is for the target device (True) or source device (False).
            max_num_shots: Maximum number of shots to include in the dataset.
            min_shot_id: Minimum shot ID to include in the dataset.
            raw_ds_dir: Directory to save the tensorized dataset.
            processed_ds_dir: Directory to save the processed dataset.
            scratch_dir: Directory to place the final datasets for training.
            extrapolate: Extrapolation for the test set, either "chronological" or "performance".
            cleanup: Clean up intermediate datasets. Options: "raw", "processed", "scratch".
            data_cluster_only: Run only the data cluster workflow (raw -> processed).
            compute_cluster_only: Run only the compute cluster workflow (processed -> scratch).
        """
        workflow = TCVDataWorkflow(
            ds_name=ds_name,
            is_target=is_target,
            max_num_shots=max_num_shots,
            min_shot_id=min_shot_id,
            raw_ds_dir=raw_ds_dir,
            processed_ds_dir=processed_ds_dir,
            scratch_dir=scratch_dir,
            extrapolate=extrapolate,
        )

        self._run_workflow(workflow, cleanup, data_cluster_only, compute_cluster_only)

    def cmod(
        self,
        ds_name: str,
        is_target: bool = True,
        max_num_shots: int = MAX_NUM_SHOTS_DEFAULT,
        min_shot_id: int = 1050204013,
        raw_ds_dir: str = RAW_DS_DIR_DEFAULT,
        processed_ds_dir: str = PROCESSED_DS_DIR_DEFAULT,
        scratch_dir: str = SCRATCH_DIR_DEFAULT,
        extrapolate: str = "chronological",
        cleanup: str | None = None,
        data_cluster_only: bool = False,
        compute_cluster_only: bool = False,
    ):
        """Create a C-Mod dataset using CMODDataWorkflow.

        Args:
            ds_name: Name of the dataset (e.g., "cmod_test", "cmod_1000").
            is_target: Whether this workflow is for the target device (True) or source device (False).
            max_num_shots: Maximum number of shots to include in the dataset.
            min_shot_id: Minimum shot ID to include in the dataset.
            raw_ds_dir: Directory to save the tensorized dataset.
            processed_ds_dir: Directory to save the processed dataset.
            scratch_dir: Directory to place the final datasets for training.
            extrapolate: Extrapolation for the test set, either "chronological" or "performance".
            cleanup: Clean up intermediate datasets. Options: "raw", "processed", "scratch".
            data_cluster_only: Run only the data cluster workflow (raw -> processed).
            compute_cluster_only: Run only the compute cluster workflow (processed -> scratch).
        """
        workflow = CMODDataWorkflow(
            ds_name=ds_name,
            is_target=is_target,
            max_num_shots=max_num_shots,
            min_shot_id=min_shot_id,
            raw_ds_dir=raw_ds_dir,
            processed_ds_dir=processed_ds_dir,
            scratch_dir=scratch_dir,
            extrapolate=extrapolate,
        )

        self._run_workflow(workflow, cleanup, data_cluster_only, compute_cluster_only)

    def _run_workflow(self, workflow, cleanup: str | None, data_cluster_only: bool, compute_cluster_only: bool):
        """Run the workflow with the specified options."""
        if cleanup:
            logger.info(f"Cleaning up {cleanup} datasets")
            workflow.cleanup(cleanup)

        if data_cluster_only:
            logger.info("Running data cluster workflow only")
            workflow.run_data_cluster()
        elif compute_cluster_only:
            logger.info("Running compute cluster workflow only")
            workflow.run_compute_cluster()
        else:
            logger.info("Running full workflow")
            workflow.run()

        logger.info("Dataset creation completed!")


def main():
    """Main entry point for the CLI."""
    fire.Fire(DatasetCLI)


if __name__ == "__main__":
    main()
