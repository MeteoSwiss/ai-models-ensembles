# FCN3 (FourCastNet v3) model container for GH200 (aarch64).
# Base: NGC PyTorch 25.12
#
# Same torch-harmonics fix as sfno, plus extra build deps for makani.
FROM nvcr.io/nvidia/pytorch:25.12-py3

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /workspace/ai-models-ensembles

ENV PIP_CONSTRAINT= \
    UV_NO_CACHE=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio

# Build deps that fcn3/makani needs to compile from source
RUN uv pip install --system --break-system-packages \
    hatchling wheel_stub py-cpuinfo setuptools-scm Cython

RUN uv pip install --system --break-system-packages --no-build-isolation \
    "earth2studio[fcn3,data]@git+https://github.com/NVIDIA/earth2studio.git@main"

RUN uv pip install --system --break-system-packages nvidia-ml-py
# physicsnemo uses wp.context.Device which was removed in warp-lang 1.13
RUN uv pip install --system --break-system-packages "warp-lang<1.13"

# Rebuild torch-harmonics with CUDA extensions against NGC torch
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA_EXTENSION=1
RUN uv pip install --system --break-system-packages scipy sympy networkx
RUN uv pip install --system --break-system-packages --no-deps --no-build-isolation --reinstall \
    "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git"
RUN uv pip install --system --break-system-packages --no-deps --no-build-isolation --reinstall \
    "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"

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
