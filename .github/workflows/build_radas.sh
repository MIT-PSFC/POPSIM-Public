#!/bin/bash

# Run radas except in CI environment.
ATOMIC_DATA_PATH=$(git rev-parse --show-toplevel)/popsim/atomic_data

# Run the radas command using the obtained path within the Poetry environment
echo "Running radas. This might take a while (sorry)!"
poetry run radas -d "$ATOMIC_DATA_PATH" --verbose
echo "Moving the radas output files to $ATOMIC_DATA_PATH..."
mv ${ATOMIC_DATA_PATH}/output/*.nc "${ATOMIC_DATA_PATH}/"