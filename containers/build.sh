#!/usr/bin/env bash
# Build a per-model container image for GH200 (aarch64).
#
# Pipeline:
#   1. podman build  - linux/arm64 image from containers/<model>.dockerfile
#   2. enroot import - convert to .sqsh for pyxis
#
# Usage:
#   ./containers/build.sh graphcast        # -> graphcast.sqsh
#   ./containers/build.sh atlas            # -> atlas.sqsh
#   OUTPUT=/path/to/sfno.sqsh ./containers/build.sh sfno
#
# Env vars:
#   OUTPUT     (default: <model>.sqsh)
#   PLATFORM   (default: linux/arm64)

set -euo pipefail
IFS=$'\n\t'

MODEL="${1:?Usage: build.sh <model>  (aurora|graphcast|sfno|fcn3|atlas)}"
DOCKERFILE="containers/${MODEL}.dockerfile"
IMAGE_TAG="ai-ens-${MODEL}:latest"
OUTPUT="${OUTPUT:-${MODEL}.sqsh}"
PLATFORM="${PLATFORM:-linux/arm64}"

cd "$(dirname "$0")/.."

if [[ ! -f "$DOCKERFILE" ]]; then
    echo "ERROR: $DOCKERFILE not found. Available:" >&2
    ls containers/*.dockerfile 2>/dev/null >&2
    exit 1
fi

if ! command -v podman >/dev/null 2>&1; then
    echo "podman not found; please load it via your module system or install it." >&2
    exit 1
fi
if ! command -v enroot >/dev/null 2>&1; then
    echo "enroot not found; required to convert podman image -> .sqsh for pyxis." >&2
    exit 1
fi

echo "==> podman build (${PLATFORM}) ${DOCKERFILE} -> ${IMAGE_TAG}"
podman build --platform "${PLATFORM}" -t "${IMAGE_TAG}" -f "${DOCKERFILE}" .

echo "==> enroot import -> ${OUTPUT}"
rm -f "${OUTPUT}"
enroot import -o "${OUTPUT}" "podman://localhost/${IMAGE_TAG}"

echo "Built ${OUTPUT}. Use with srun --container-image=$(realpath "${OUTPUT}")"
