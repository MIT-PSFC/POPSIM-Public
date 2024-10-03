#!/bin/bash


ATOMIC_DATA_PATH=$(git rev-parse --show-toplevel)/popsim/atomic_data

echo "Running radas. This might take a while (sorry)!"
poetry run radas -d "$ATOMIC_DATA_PATH" --verbose

echo "Moving the radas output files to $ATOMIC_DATA_PATH..."
mv ${ATOMIC_DATA_PATH}/output/*.nc "${ATOMIC_DATA_PATH}/"
cp ${ATOMIC_DATA_PATH}/*.nc ./submodules/cfspopcon/cfspopcon/atomic_data/