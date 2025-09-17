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

# Check if uv is installed and its version. Install if not present or update if version is less than the minimum required version.
echo "Checking for uv installation..."
if command -v uv >/dev/null 2>&1; then
    UV_VERSION=$(uv --version | awk '{print $2}')
    MINIMUM_VERSION="0.4.0"
    if [ "$(printf '%s\n' "$MINIMUM_VERSION" "$UV_VERSION" | sort -V | head -n1)" != "$MINIMUM_VERSION" ]; then
        echo "uv version $UV_VERSION is less than the minimum required version $MINIMUM_VERSION."
        NEED_UV_INSTALL=y
    else
        echo "uv version $UV_VERSION meets the requirement."
        NEED_UV_INSTALL=n
    fi
else
    echo "uv is not installed. Need to install uv."
    NEED_UV_INSTALL=y
fi

# Prompt the user to install uv if not present or update if version is less than the minimum required version.
install_uv=$(prompt_user "Do you want to install uv? (y/n):" "y")
if [[ "$install_uv" =~ ^[yY] ]]; then
    echo "Installing the latest uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Navigate to the root of the repository
cd "$(dirname "$0")"

# Check if VIRTUAL_ENV is set. We should exit the virtual environment before doing a uv install.
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment detected. Unsetting VIRTUAL_ENV."
    unset VIRTUAL_ENV
fi

# Install dependencies
echo "Installing dependencies with uv..."
uv sync
uv run setup-radas # Install radas

# Optional installations
install_gpu=$(prompt_user "Do you want to install with GPU support? (y/n):" "$INSTALL_GPU")
# If install_gpu starts with y or Y then install.
if [[ "$install_gpu" =~ ^[yY] ]]; then
    echo "Installing with GPU support..."
    uv sync --group gpu
fi

install_dev=$(prompt_user "Do you want to install development dependencies? (y/n):" "$INSTALL_DEV")
# If install_dev starts with y or Y then install.
if [[ "$install_dev" =~ ^[yY] ]]; then
    echo "Installing development dependencies..."
    uv sync --group dev
    uv run pre-commit install # Install pre-commit hooks
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
echo "Remember to use 'uv run' to execute commands in the virtual environment. For more information, refer to the uv documentation."