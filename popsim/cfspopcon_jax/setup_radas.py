import os
import subprocess

import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()


def run_radas():
    print("Running radas to download atomic data...")
    subprocess.run(["radas", "-d", "./atomic_data"], check=True)


if __name__ == "__main__":
    run_radas()
