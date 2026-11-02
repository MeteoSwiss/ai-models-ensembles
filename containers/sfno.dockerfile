# SFNO model container for GH200 (aarch64).
# Base: NGC PyTorch 25.12
#
# Key: torch-harmonics must be rebuilt from source against NGC's torch
# to get CUDA extensions. The --no-deps --no-build-isolation --reinstall
# pattern prevents uv from pulling a CPU-only PyTorch from PyPI.
# Git dependencies are pinned to the commits found in the built image; the full
# package freeze of that image is containers/locks/sfno.txt (python containers/freeze_sqsh.py).
FROM nvcr.io/nvidia/pytorch:25.12-py3

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /workspace/ai-models-ensembles

ENV PIP_CONSTRAINT= \
    UV_NO_CACHE=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio

RUN uv pip install --system --break-system-packages hatchling

RUN uv pip install --system --break-system-packages --no-build-isolation \
    "earth2studio[sfno,data]@git+https://github.com/NVIDIA/earth2studio.git@cbcb0ec0e91059ecacba7714ca42946531cbc0ad"

# physicsnemo uses wp.context.Device which was removed in warp-lang 1.13
RUN uv pip install --system --break-system-packages "warp-lang<1.13"

# Rebuild torch-harmonics and makani with CUDA extensions against NGC torch
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA_EXTENSION=1
RUN uv pip install --system --break-system-packages --no-deps --no-build-isolation --reinstall \
    "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@8f08a221ce50593af1202ef9833de0bde178efe1"
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
