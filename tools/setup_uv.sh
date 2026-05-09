#!/usr/bin/env bash
# Minimal uv + venv setup (Linux) with JAX CUDA 12 and PyTorch, no source builds.
# - Creates .venv with Python 3.11 (symlinked to $STORE on Clariden for perf)
# - Installs project in editable mode and runtime deps from PyPI
# - Installs JAX GPU on NVIDIA aarch64 (CUDA 12), otherwise CPU JAX
# - Installs PyTorch from official wheels (aarch64 via NVIDIA index, x86_64 cu124 or CPU)
#
# Usage:
#   bash tools/setup_uv.sh              # create or reuse .venv
#   RECREATE=1 bash tools/setup_uv.sh   # wipe and rebuild from scratch

set -euo pipefail
IFS=$'\n\t'

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYVER=${PYVER:-3.11}
RECREATE=${RECREATE:-0}

# Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and activate.
# On Clariden the venv lives on $STORE (capstor) for performance and is
# symlinked into the repo as .venv. RECREATE=1 wipes the target dir contents
# but preserves the symlink.
uv python install "$PYVER"

VENV_DIR=".venv"
if [[ "$RECREATE" == "1" && -e "$VENV_DIR" ]]; then
  # Follow symlink: delete contents of the real directory, not the link itself
  REAL_DIR=$(readlink -f "$VENV_DIR" 2>/dev/null || echo "$VENV_DIR")
  echo "Removing existing venv at $REAL_DIR (RECREATE=1)"
  rm -rf "$REAL_DIR"
  # If .venv was a symlink, the target is gone; remove the dangling link
  [[ -L "$VENV_DIR" ]] && rm -f "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  # On Clariden, place the venv on capstor and symlink into the repo
  if [[ -n "${STORE:-}" ]]; then
    VENV_STORE="$STORE/$USER/.venv-ai-models-ensembles"
    echo "Creating venv on capstor: $VENV_STORE"
    uv venv "$VENV_STORE" -p "$PYVER"
    ln -sfn "$VENV_STORE" "$VENV_DIR"
  else
    uv venv "$VENV_DIR" -p "$PYVER"
  fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

ARCH=$(uname -m)
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1; then HAS_NVIDIA=1; fi

# Install PyTorch first (flash-attn and other extras need it at build time)
echo "Installing PyTorch"
if [[ "$ARCH" == "aarch64" ]]; then
  uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.4.*'
else
  if [[ $HAS_NVIDIA -eq 1 ]]; then
    uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.4.*'
  else
    uv pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.4.*'
  fi
fi

# Install project in editable mode (earth2studio base + all pure-Python deps)
echo "Installing project base"
uv pip install -e .

# Install earth2studio model extras that work on this platform.
# flash-attn (aifs) needs torch at build time, so --no-build-isolation.
# On aarch64: sfno/fcn3 need makani (no aarch64 wheel), pangu/fuxi need
# onnxruntime-gpu (no aarch64 wheel) -- skip them.
echo "Installing earth2studio model extras"
if [[ "$ARCH" == "aarch64" ]]; then
  uv pip install --no-build-isolation 'earth2studio[graphcast,gencast,aifs]'
else
  uv pip install --no-build-isolation 'earth2studio[graphcast,fcn,fcn3,sfno,gencast,pangu,fuxi,aifs]'
fi

echo "Installing JAX"
if [[ "$ARCH" == "aarch64" && $HAS_NVIDIA -eq 1 ]]; then
  uv pip install 'jax[cuda12]'
else
  uv pip install 'jax'
fi

echo "Setup complete."
