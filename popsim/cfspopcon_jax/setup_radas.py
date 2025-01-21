import os
import subprocess

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()


def run_radas():
    output_dir = "./atomic_data/output"

    # Check if output directory exists and contains files
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Atomic data already exists in '{output_dir}'. Skipping download.")
        return

    print("Running radas to download atomic data...")
    subprocess.run(["radas", "-d", "./atomic_data"], check=True)


if __name__ == "__main__":
    run_radas()
