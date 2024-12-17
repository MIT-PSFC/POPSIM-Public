#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update git submodules
echo "Updating git submodules..."
git submodule update --init --recursive

# Function to prompt the user or use default in CI
prompt_user() {
    local prompt_message="$1"
    local install_test_value="$2"
    local user_input
    
    if [ -n "$INSTALL_TEST" ]; then
        # In CI environment, use default value
        user_input="$install_test_value"
    else
        # Interactive prompt
        read -p "$prompt_message " user_input
    fi
    echo "$user_input"
}

# Check if Poetry is installed and its version. Install if not present or update if version is less than the minimum required version.
echo "Checking for Poetry installation..."
if command -v poetry >/dev/null 2>&1; then
    POETRY_VERSION=$(poetry --version | awk '{print $3}')
    MINIMUM_VERSION="1.6.1"
    if [ "$(printf '%s\n' "$MINIMUM_VERSION" "$POETRY_VERSION" | sort -V | head -n1)" != "$MINIMUM_VERSION" ]; then 
        echo "Poetry version $POETRY_VERSION is less than the minimum required version $MINIMUM_VERSION."
        NEED_POETRY_INSTALL=y
    else
        echo "Poetry version $POETRY_VERSION meets the requirement."
        NEED_POETRY_INSTALL=n
    fi
else
    echo "Poetry is not installed. Need to install Poetry."
    NEED_POETRY_INSTALL=y
fi

# Prompt the user to install Poetry if not present or update if version is less than the minimum required version.
install_poetry=$(prompt_user "Do you want to install Poetry? (y/n):" "y")
if [[ "$install_poetry" =~ ^[yY] ]]; then
    echo "Installing the latest Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
fi


# Ensure the correct version of Poetry is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Navigate to the root of the repository
cd "$(dirname "$0")"

# Check if VIRTUAL_ENV is set. We should exit the virtual environment before doing a poetry install.
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment detected. Unsetting VIRTUAL_ENV."
    unset VIRTUAL_ENV
fi

# Install dependencies
echo "Installing dependencies with Poetry..."
poetry install


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
    poetry run pre-commit install # Install pre-commit hooks
    poetry run setup-radas # Install radas
fi

# Check for git lfs and install if not present
echo "Checking for git LFS..."
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git LFS is not installed.."
    install_git_lfs=$(prompt_user "Do you want to install git LFS? (y/n):" "y")

    if [[ "$install_git_lfs" =~ ^[yY] ]]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if command -v apt >/dev/null 2>&1; then
                sudo apt install git-lfs
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install git-lfs
            else
                echo "Please install git LFS manually."
                exit 1
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew >/dev/null 2>&1; then
                brew install git-lfs
            else
                echo "Please install Homebrew and then install git LFS."
                exit 1
            fi
        else
            echo "Unsupported OS. Please install git LFS manually."
            exit 1
        fi
    else
        echo "Please install git LFS manually."
    fi
fi

echo "Initializing and pulling with git LFS..."
git lfs install
git lfs pull

echo "Installation complete."
echo "Remember to use 'poetry shell' or prefix your commands with 'poetry run'. For more information, refer to the Poetry documentation."