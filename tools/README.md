# Development Tools

This directory contains setup scripts, testing utilities, and monitoring tools for the `ai-models-ensembles` repository.

## Setup Scripts

### setup_uv.sh

Initial environment setup using uv package manager.

**Usage:**

```bash
bash tools/setup_uv.sh
source .venv/bin/activate
```

**What it does:**

- Creates Python 3.10 virtual environment
- Installs all dependencies via uv
- Configures JAX and PyTorch for GPU support
- Handles platform-specific installations (ARM/x86_64)

### check_gpu.py

Verify GPU availability and configuration.

**Usage:**

```bash
python tools/check_gpu.py
```

**What it checks:**

- CUDA availability
- GPU device count and names
- GPU memory
- Driver versions

## Validation & Testing

### validate.sh

Comprehensive environment and configuration validation.

**Usage:**

```bash
bash ./tools/validate.sh
```

**What it validates:**

- Configuration variables (OUTPUT_DIR, DATE_TIME, MODEL_NAME, etc.)
- Python version (3.10.x recommended)
- Required Python packages
- External tools (ai-models, ImageMagick, GRIB tools)
- ECMWF credentials and MARS connectivity
- Directory permissions

**Expected output:** "Validation completed; please review any warnings above."

### test_basic_functionality.py

Comprehensive functionality test without requiring data or GPU.

**Usage:**

```bash
source .venv/bin/activate
python tools/test_basic_functionality.py
```

**Tests performed:**

- Python package imports
- Module imports (CLI, preprocessing, plotting, etc.)
- CLI command accessibility
- Dependency versions
- ai-models installation
- Configuration file existence

**Expected output:** All tests pass with ✓ marks

### run_minimal_test.sh

Quick end-to-end validation.

**Usage:**

```bash
./tools/run_minimal_test.sh
```

**Steps executed:**

- Activate virtual environment
- Load configuration
- Create output directories
- Test CLI commands
- List available models
- Generate fields file

**Expected output:** Summary showing all checks passing

## Workflow Monitoring

### check_workflow_status.sh

Monitor workflow progress and completion status.

**Usage:**

```bash
./tools/check_workflow_status.sh
```

**Information displayed:**

- Current configuration settings
- Data download status (ERA5, IFS ensemble, IFS control)
- Model fields file status
- Ensemble member count and completion
- Forecast output status
- Verification and plots status
- Recommended next step

**Example output:**

```
==========================================
Workflow Status Check
==========================================

Configuration:
  DATE_TIME:          201801010000
  MODEL_NAME:         graphcast
  NUM_MEMBERS:        50
  ...

Step 1: Initial Conditions (ERA5)
✓ Initial field GRIB
  Location: /path/to/init_field.grib
  Size: 45M

...

Status: Complete! ✓
```

## Quick Reference

```bash
# First-time setup
bash tools/setup_uv.sh && source .venv/bin/activate

# Validate environment
bash ./tools/validate.sh

# Test installation
python tools/test_basic_functionality.py

# Monitor progress
./tools/check_workflow_status.sh
```

For a complete step-by-step workflow example, see **[QUICKSTART_TEST.md](QUICKSTART_TEST.md)**.

## Troubleshooting

- **Setup fails**: Ensure uv is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Missing packages**: Reinstall with `uv pip install -e .` after activating environment
- **No GPUs detected**: Verify with `nvidia-smi` and check driver version >= 525
- **Status shows no data**: Ensure you've sourced `scripts/config.sh` and run download steps

For workflow execution issues, see [scripts/README.md](../scripts/README.md).

## See Also

- [Main README](../README.md) - Full repository documentation
- [scripts/config.sh](../scripts/config.sh) - Configuration settings
- [scripts/](../scripts/) - Workflow execution scripts (submit_*.sh)
