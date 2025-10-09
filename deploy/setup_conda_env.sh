#!/bin/bash
# SpatX Conda Environment Setup Script for Lab PC
# No sudo access required - pure conda deployment

set -e  # Exit on error

echo "🚀 Setting up SpatX Conda Environment..."

# Configuration
ENV_NAME="spatx"
PYTHON_VERSION="3.11"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found! Please ensure conda is installed and in PATH"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists. Removing..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new conda environment
echo "📦 Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo "🔧 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Install PyTorch with CUDA support (if available)
echo "🔥 Installing PyTorch..."
# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU detected, installing PyTorch with CUDA 11.8"
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "   No GPU detected, installing CPU-only PyTorch"
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

# Install core dependencies via conda (faster and more reliable than pip)
echo "📦 Installing core dependencies via conda..."
conda install -y -c conda-forge \
    numpy \
    pandas \
    pillow \
    scikit-learn \
    scipy \
    matplotlib \
    fastapi \
    uvicorn \
    python-jose \
    passlib \
    python-multipart \
    sqlalchemy \
    bcrypt

# Install remaining dependencies via pip
echo "📦 Installing additional dependencies via pip..."
pip install --no-cache-dir \
    cryptography \
    pydantic \
    pydantic-settings \
    python-jose[cryptography] \
    openslide-python

echo ""
echo "✅ Conda environment '${ENV_NAME}' created successfully!"
echo ""
echo "📋 Environment Info:"
conda info
echo ""
echo "📦 Installed packages:"
conda list | head -20
echo "   ... (truncated)"
echo ""
echo "🎯 To activate this environment, run:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "🚀 To start SpatX backend, run:"
echo "   conda activate ${ENV_NAME}"
echo "   python app_enhanced.py"


