# ai-models-ensembles container for earth2studio inference on GH200 nodes.
#
# Builds an aarch64 image based on NVIDIA's PyTorch container, installs uv, and
# editable-installs this repo with all earth2studio model extras enabled.
#
# Build:
#   podman build --platform linux/arm64 -t ai-ens:latest -f containers/Dockerfile .
#   enroot import -o ai-ens.sqsh podman://localhost/ai-ens:latest
#
# Run (Slurm + pyxis):
#   srun --container-image=./ai-ens.sqsh -- ai-ens infer ...

ARG NVIDIA_PYTORCH_TAG=25.12-py3
FROM nvcr.io/nvidia/pytorch:${NVIDIA_PYTORCH_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio

RUN apt-get update && apt-get install -y --no-install-recommends \
        gettext-base \
        git \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# uv for fast resolves
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/ai-models-ensembles
COPY pyproject.toml ./
COPY ai_models_ensembles ./ai_models_ensembles
COPY config ./config

# Editable install pulls earth2studio + selected model extras + swissclim.
RUN uv pip install --system -e .

# Pre-create cache dir so EARTH2STUDIO_CACHE is writable in the container.
RUN mkdir -p ${EARTH2STUDIO_CACHE}

ENTRYPOINT ["ai-ens"]
CMD ["--help"]
