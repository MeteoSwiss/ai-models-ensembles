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

# uv for fast resolves (NGC 25.12+ already ships uv; update if present)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sfn /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace/ai-models-ensembles
COPY pyproject.toml ./
COPY ai_models_ensembles ./ai_models_ensembles
COPY config ./config
COPY tools ./tools
COPY tests ./tests
COPY scripts ./scripts

# NGC 25.12 marks its Python as externally managed; override that.
ENV UV_SYSTEM_PACKAGES=1 UV_BREAK_SYSTEM_PACKAGES=1

# Step 1: Install natten first without build isolation (needs NGC torch).
RUN uv pip install --system --break-system-packages --no-build-isolation natten

# Step 2: Editable install of the project (normal isolation for everything else).
RUN uv pip install --system --break-system-packages -e .

# Step 3: Fix dependency issues:
# - warp-lang<1.13: physicsnemo 2.0 uses wp.context.Device (removed in 1.13)
# - graphcast: PyPI "graphcast" is a wrong package (graph-DB tool); replace
#   with DeepMind's graphcast from GitHub
# - chex: missing transitive dep for graphcast model loading
RUN uv pip install --system --break-system-packages "warp-lang<1.13" \
    && uv pip uninstall --system graphcast \
    && uv pip install --system --break-system-packages \
        chex "graphcast @ git+https://github.com/google-deepmind/graphcast"

# Pre-create cache dir so EARTH2STUDIO_CACHE is writable in the container.
RUN mkdir -p ${EARTH2STUDIO_CACHE}

ENTRYPOINT ["ai-ens"]
CMD ["--help"]
