#!/bin/bash

# Run radas except in CI environment.
if [ -z "$CI" ]; then
    echo "Retrieving ATOMIC_DATA_PATH from popsim module..."
    ATOMIC_DATA_PATH=$(poetry run python -c "from popsim import ATOMIC_DATA_PATH; print(ATOMIC_DATA_PATH)")
    
    # Run the radas command using the obtained path within the Poetry environment
    echo "Running radas. This might take a while (sorry)!"
    poetry run radas -d "$ATOMIC_DATA_PATH" --verbose
    echo "Moving the radas output files to $ATOMIC_DATA_PATH..."
    mv ${ATOMIC_DATA_PATH}/output/*.nc "${ATOMIC_DATA_PATH}/"
else
    echo "Skipping radas execution in CI environment."
fi
