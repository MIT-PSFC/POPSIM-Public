import glob
import os
from concurrent.futures import ThreadPoolExecutor

import fire
from tqdm import tqdm

from popsim.interfaces.meq2xarray.meq2xarray import tcv_db_to_xr, tcv_fbt_to_xr


def mat_to_xr_cli(paths: str, output_dir: str, file_type: str, use_zarr: bool = False, workers: int = 1):
    """Convert .mat files to xarray format.

    Args:
        paths: Glob pattern or path to .mat files
        output_dir: Output directory for converted files
        file_type: Type of file ('db' or 'fbt')
        use_zarr: Save files in Zarr format instead of NetCDF
        workers: Number of workers for parallel processing
    """
    if isinstance(paths, str):
        paths = glob.glob(paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    workers = min(workers, len(paths))

    if workers > 1 and not use_zarr:
        raise ValueError("Parallel processing is only supported with Zarr output.")

    if file_type == "db":
        mat_path_to_data_fn = tcv_db_to_xr
    elif file_type == "fbt":
        mat_path_to_data_fn = tcv_fbt_to_xr
    else:
        raise ValueError("File type must be either 'db' or 'fbt'.")

    def process_file(mat_path):
        try:
            # Convert .mat file to xr.DataTree or xr.Dataset
            dt_or_ds = mat_path_to_data_fn(mat_path)

            # Generate output path
            filename = os.path.basename(mat_path).replace(".mat", ".zarr" if use_zarr else ".nc")
            output_path = os.path.join(output_dir, filename)

            # Save the data in the desired format
            if use_zarr:
                dt_or_ds.to_zarr(output_path)
            else:
                dt_or_ds.to_netcdf(output_path)

            print(f"Converted and saved: {mat_path} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {mat_path}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_file, paths), total=len(paths), desc="Converting .mat files"))


if __name__ == "__main__":
    fire.Fire()
