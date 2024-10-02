#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update git submodules
echo "Updating git submodules..."
git submodule update --init --recursive

# Check if Poetry is installed and its version. Install if not present or update if version is less than the minimum required version.
echo "Checking for Poetry installation..."
if command -v poetry >/dev/null 2>&1; then
    POETRY_VERSION=$(poetry --version | awk '{print $3}')
    MINIMUM_VERSION="1.6.1"
    if [ "$(printf '%s\n' "$MINIMUM_VERSION" "$POETRY_VERSION" | sort -V | head -n1)" != "$MINIMUM_VERSION" ]; then 
        echo "Poetry version $POETRY_VERSION is less than the minimum required version $MINIMUM_VERSION."
        echo "Updating Poetry to the latest version..."
        curl -sSL https://install.python-poetry.org | python3 -
    else
        echo "Poetry version $POETRY_VERSION meets the requirement."
    fi
else
    echo "Poetry is not installed. Installing the latest Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Ensure the correct version of Poetry is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Navigate to the root of the repository
cd "$(dirname "$0")"

# Install dependencies
echo "Installing dependencies with Poetry..."
poetry install


# Function to prompt the user or use default in CI
prompt_user() {
    local prompt_message="$1"
    local default_value="$2"
    local user_input
    
    if [ -n $CI ]; then
        # In CI environment, use default value
        user_input="$default_value"
    else
        # Interactive prompt
        read -p "$prompt_message " user_input
    fi
    echo "$user_input"
}

# Optional installations
install_gpu=$(prompt_user "Do you want to install with GPU support? (y/n):" "$INSTALL_GPU")
# If install_gpu starts with y or Y then install.
if [[ "$install_gpu" =~ ^[yY] ]]; then
    echo "Installing with GPU support..."
    poetry install --with gpu
fi

install_dev=$(prompt_user "Do you want to install development dependencies? (y/n):" "$INSTALL_DEV")
# If install_dev starts with y or Y then install.
if [[ "$install_dev" =~ ^[yY] ]]; then
    echo "Installing development dependencies..."
    poetry install --with dev
    pre-commit install # Install pre-commit hooks
fi

./build_radas.sh

# Check for git lfs and install if not present
echo "Checking for Git LFS..."
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Git LFS is not installed. Installing Git LFS..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt >/dev/null 2>&1; then
            sudo apt install git-lfs
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install git-lfs
        else
            echo "Please install Git LFS manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install git-lfs
        else
            echo "Please install Homebrew and then install Git LFS."
            exit 1
        fi
    else
        echo "Unsupported OS. Please install Git LFS manually."
        exit 1
    fi
fi

echo "Initializing and pulling with Git LFS..."
git lfs install
git lfs pull

echo "Installation complete."
echo "Remember to use 'poetry shell' or prefix your commands with 'poetry run'. For more information, refer to the Poetry documentation."
