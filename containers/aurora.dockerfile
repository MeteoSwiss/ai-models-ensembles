# Aurora model container for GH200 (aarch64).
# Base: NGC PyTorch 25.12
FROM nvcr.io/nvidia/pytorch:25.12-py3

COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

WORKDIR /workspace/ai-models-ensembles

ENV PIP_CONSTRAINT= \
    UV_NO_CACHE=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    EARTH2STUDIO_CACHE=/workspace/.cache/earth2studio

RUN uv pip install --system --break-system-packages hatchling
RUN uv pip install --system --break-system-packages --no-build-isolation \
    "earth2studio[aurora,data,perturbation]@git+https://github.com/NVIDIA/earth2studio.git@main"

# Install this project (editable so mounted code at runtime takes precedence)
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
