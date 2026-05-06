#!/usr/bin/env bash
# Build the ai-models-ensembles container image for GH200 (aarch64).
#
# Pipeline (mirrors earthPip's pattern):
#   1. podman build  - linux/arm64 image
#   2. enroot import - convert to a .sqsh squashfs for pyxis
#
# Usage:
#   ./containers/build.sh                       # tag ai-ens:latest, sqsh -> ./ai-ens.sqsh
#   IMAGE_TAG=ai-ens:dev OUTPUT=foo.sqsh ./containers/build.sh
#
# Env vars:
#   IMAGE_TAG  (default: ai-ens:latest)
#   OUTPUT     (default: ai-ens.sqsh)
#   PLATFORM   (default: linux/arm64)
#   DOCKERFILE (default: containers/Dockerfile)

set -euo pipefail
IFS=$'\n\t'

IMAGE_TAG=${IMAGE_TAG:-ai-ens:latest}
OUTPUT=${OUTPUT:-ai-ens.sqsh}
PLATFORM=${PLATFORM:-linux/arm64}
DOCKERFILE=${DOCKERFILE:-containers/Dockerfile}

cd "$(dirname "$0")/.."

if ! command -v podman >/dev/null 2>&1; then
    echo "podman not found; please load it via your module system or install it." >&2
    exit 1
fi
if ! command -v enroot >/dev/null 2>&1; then
    echo "enroot not found; required to convert podman image -> .sqsh for pyxis." >&2
    exit 1
fi

echo "==> podman build (${PLATFORM}) -> ${IMAGE_TAG}"
podman build --platform "${PLATFORM}" -t "${IMAGE_TAG}" -f "${DOCKERFILE}" .

echo "==> enroot import -> ${OUTPUT}"
rm -f "${OUTPUT}"
enroot import -o "${OUTPUT}" "podman://localhost/${IMAGE_TAG}"

echo "Built ${OUTPUT}. Use with srun --container-image=$(realpath "${OUTPUT}")"
