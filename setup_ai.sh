#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- Starting TAQYIM Environment Setup ---"

# 1. Initialize Conda for this script session
# Detects your conda installation automatically
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# 2. Create the environment (removes old one if exists to start fresh)
echo "Step 1: Creating Conda environment (Python 3.11)..."
conda create --name taqyim_env python=3.11 -y

# 3. Activate the environment
conda activate taqyim_env

# 4. Install PyTorch 2.5.1 with CUDA 12.4 support
echo "Step 2: Installing PyTorch (this is a large download)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install Unsloth
echo "Step 3: Installing Unsloth from GitHub..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 6. Install specific optimized dependencies for RTX 3050
echo "Step 4: Installing xformers and PEFT..."
pip install --no-deps "xformers<0.0.30" trl peft accelerate bitsandbytes

# 7. Install data processing tools
echo "Step 5: Installing data tools (Pandas, TQDM, Datasets)..."
pip install pandas tqdm datasets

echo "--- ALL DONE! ---"
echo "To start working, run: conda activate taqyim_env"
