#!/bin/bash

# Exit immediately if a command fails
set -e

# Define colors
GREEN='\033[0;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create the conda environment
echo -e "${BLUE}Creating conda environment 'pacs' with Python 3.10...${NC}"
conda create --name pacs python=3.10 -y

# Activate the environment
echo -e "${BLUE}Activating the 'pacs' environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pacs

# Install packages from conda-forge
echo -e "${GREEN}Installing core scientific packages from conda-forge...${NC}"
conda install -c conda-forge fenics-dolfinx mpich pyvista numpy scipy matplotlib pandas tensorflow ipython compilers -y

# Install GMSH and Python bindings
echo -e "${GREEN}Installing GMSH and Python bindings...${NC}"
conda install -c conda-forge gmsh python-gmsh -y

# Install Python packages with pip
echo -e "${YELLOW}Installing additional Python packages with pip...${NC}"
pip install scikit-learn meshio

# Install CUDA toolkit
echo -e "${GREEN}Installing CUDA toolkit...${NC}"
conda install cuda-cudart cuda-version=12 -y

# Final message
echo -e "${BLUE}✅ Environment setup complete! Activate it anytime with:${NC}"
echo -e "${YELLOW}conda activate pacs${NC}"


echo "[INFO] Searching for libdevice.10.bc in current Conda environment..."

# Locate libdevice.10.bc in current conda env
LIBDEVICE_PATH=$(find "$CONDA_PREFIX" -name libdevice.10.bc | head -n 1)

if [ -z "$LIBDEVICE_PATH" ]; then
    echo "[ERROR] Could not find libdevice.10.bc in your conda environment."
    exit 1
fi

# Check if symlink exists and update if needed
if [ -L ./libdevice.10.bc ]; then
    CURRENT_TARGET=$(readlink ./libdevice.10.bc)
    if [ "$CURRENT_TARGET" != "$LIBDEVICE_PATH" ]; then
        echo "[INFO] Symlink exists but points to: $CURRENT_TARGET"
        echo "[INFO] Updating symlink to point to: $LIBDEVICE_PATH"
        ln -sf "$LIBDEVICE_PATH" ./libdevice.10.bc
        echo "[INFO] Symlink updated."
    else
        echo "[INFO] Symlink already exists and is correct."
    fi
elif [ -e ./libdevice.10.bc ]; then
    echo "[WARNING] ./libdevice.10.bc exists but is not a symlink. Backing it up."
    mv ./libdevice.10.bc ./libdevice.10.bc.bak
    ln -s "$LIBDEVICE_PATH" ./libdevice.10.bc
    echo "[INFO] Created new symlink after backup."
else
    ln -s "$LIBDEVICE_PATH" ./libdevice.10.bc
    echo "[INFO] Created symlink in current directory: ./libdevice.10.bc"
fi

echo "[SUCCESS] libdevice path configured. You can now run your TensorFlow script safely."


