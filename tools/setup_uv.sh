#!/usr/bin/env bash
# Minimal uv + venv setup (Linux) with JAX CUDA 12 and PyTorch, no source builds.
# - Creates .venv with Python 3.10
# - Installs project in editable mode and runtime deps from PyPI
# - Installs JAX GPU on NVIDIA aarch64 (CUDA 12), otherwise CPU JAX
# - Installs PyTorch from official wheels (aarch64 via NVIDIA index, x86_64 cu124 or CPU)

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYVER=${PYVER:-3.10}
RECREATE=${RECREATE:-0}

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and activate
uv python install "$PYVER"
if [[ -d .venv && "$RECREATE" == "1" ]]; then
  echo "Removing existing .venv (RECREATE=1)"
  rm -rf .venv
fi
if [[ ! -d .venv ]]; then
  uv venv .venv -p "$PYVER"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Install project in editable mode
echo "Installing project base"
uv pip install -e .

ARCH=$(uname -m)
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then HAS_NVIDIA=1; fi

echo "Installing PyTorch"
if [[ "$ARCH" == "aarch64" ]]; then
  # Use PyTorch's cu124 wheel index (aarch64 CUDA-enabled wheels available)
  uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.4.*'
else
  if [[ $HAS_NVIDIA -eq 1 ]]; then
    # Target CUDA 12.4 build on x86_64
    uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.4.*'
  else
    uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.4.*'
  fi
fi

echo "Installing JAX"
if [[ "$ARCH" == "aarch64" && $HAS_NVIDIA -eq 1 ]]; then
  # CUDA 12 pip wheels bundle matching CUDA/cuDNN; no local toolkit required
  uv pip install 'jax[cuda12]'
else
  uv pip install 'jax'
fi

echo "Setup complete."