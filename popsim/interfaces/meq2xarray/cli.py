import glob
import os
from concurrent.futures import ThreadPoolExecutor

import click
from tqdm import tqdm

from popsim.interfaces.meq2xarray.meq2xarray import tcv_db_to_xr


@click.command()
@click.argument("paths", type=click.STRING)
@click.argument(
    "output_dir",
    type=click.Path(),
)
@click.option("--use-zarr", is_flag=True, help="Save files in Zarr format instead of NetCDF.")
@click.option("--workers", default=1, type=int, help="Number of workers for parallel processing.")
def convert_ss_mat_to_xr_cli(paths, output_dir, use_zarr, workers):
    if isinstance(paths, str):
        paths = glob.glob(paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    workers = max(workers, len(paths))

    if workers > 1 and not use_zarr:
        raise ValueError("Parallel processing is only supported with Zarr output.")

    def process_file(mat_path):
        try:
            # Convert .mat file to xr.DataTree
            dt = tcv_db_to_xr(mat_path)

            # Generate output path
            filename = os.path.basename(mat_path).replace(".mat", ".zarr" if use_zarr else ".nc")
            output_path = os.path.join(output_dir, filename)

            # Save the DataTree in the specified format
            if use_zarr:
                dt.to_zarr(output_path)
            else:
                dt.to_netcdf(output_path)

            print(f"Converted and saved: {mat_path} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {mat_path}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(process_file, paths), total=len(paths), desc="Converting .mat files"))


if __name__ == "__main__":
    convert_ss_mat_to_xr_cli()
