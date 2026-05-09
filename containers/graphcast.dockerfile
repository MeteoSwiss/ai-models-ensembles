# GraphCast model container for GH200 (aarch64).
# Base: NGC PyTorch 25.12
FROM nvcr.io/nvidia/pytorch:25.12-py3

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /workspace/ai-models-ensembles

ENV PIP_CONSTRAINT= \
    UV_NO_CACHE=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio

# Build deps that earth2studio[graphcast] needs to compile from source
RUN uv pip install --system --break-system-packages \
    hatchling wheel_stub py-cpuinfo setuptools-scm Cython

RUN uv pip install --system --break-system-packages --no-build-isolation \
    "earth2studio[graphcast,data]@git+https://github.com/NVIDIA/earth2studio.git@main"

# earth2studio's graphcast uses xr.Dataset(dataset) which became a TypeError
# in xarray 2026.04. Pin to avoid the breakage until upstream fixes it.
RUN uv pip install --system --break-system-packages "xarray<2026.4"

# Fix: PyPI "graphcast" is a graph-DB tool, not DeepMind's. Replace it.
# chex is a missing transitive dep for graphcast model loading.
RUN uv pip uninstall --system --break-system-packages graphcast \
    && uv pip install --system --break-system-packages \
        chex "graphcast @ git+https://github.com/google-deepmind/graphcast"

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
