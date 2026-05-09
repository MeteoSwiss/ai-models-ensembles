# Atlas model container for GH200 (aarch64).
# Base: NGC PyTorch 26.01 (needed for natten 0.21.5 compatibility)
FROM nvcr.io/nvidia/pytorch:26.01-py3

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /workspace/ai-models-ensembles

ENV PIP_CONSTRAINT= \
    UV_NO_CACHE=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio \
    EARTH2STUDIO_DTYPE="bfloat16"

# cmake needed for natten build
RUN apt-get update && apt-get install -y cmake && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system --break-system-packages \
    hatchling ninja Cython wheel_stub py-cpuinfo setuptools-scm

# Build natten from source with CUDA kernels for GH200 (SM 9.0 / Hopper)
ENV NATTEN_CUDA_ARCH="9.0"
ENV NATTEN_WITH_CUDA=1
RUN pip install --no-build-isolation --break-system-packages natten==0.21.5

# Pre-install torch-harmonics with --no-build-isolation so it can find torch
# (uv's default build isolation hides torch, breaking the CUDA extension build)
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA_EXTENSION=1
RUN uv pip install --system --break-system-packages --no-deps --no-build-isolation --reinstall torch-harmonics

RUN uv pip install --system --break-system-packages --no-build-isolation \
    "earth2studio[atlas,data]@git+https://github.com/NVIDIA/earth2studio.git@main"

# physicsnemo uses wp.context.Device which was removed in warp-lang 1.13
RUN uv pip install --system --break-system-packages "warp-lang<1.13"

# Install this project
COPY pyproject.toml ./
COPY ai_models_ensembles ./ai_models_ensembles
COPY config ./config
COPY tools ./tools
COPY tests ./tests
COPY scripts ./scripts
RUN uv pip install --system --break-system-packages --no-deps -e .

RUN mkdir -p ${EARTH2STUDIO_CACHE}
ENTRYPOINT ["ai-ens"]
CMD ["--help"]
