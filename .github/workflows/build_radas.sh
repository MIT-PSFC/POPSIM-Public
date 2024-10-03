#!/bin/bash

ATOMIC_DATA_PATH=$(git rev-parse --show-toplevel)/popsim/atomic_data

echo "Running radas. This might take a while (sorry)!"

python3 -m pip install radas==2024.8.0

python3 -m radas -d "$ATOMIC_DATA_PATH" --verbose

echo "Moving the radas output files to $ATOMIC_DATA_PATH..."
mv ${ATOMIC_DATA_PATH}/output/*.nc "${ATOMIC_DATA_PATH}/"

# TODO(allenw): remove this once cfspopcon is no longer a submodule
cp ${ATOMIC_DATA_PATH}/*.nc ./submodules/cfspopcon/cfspopcon/atomic_data/