# env/

Pinned environments outside the containers (container pins: [../containers/locks/](../containers/locks/)).
Explanation of all three environments: [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md).

| File | What |
|---|---|
| `host-requirements.txt` | exact freeze of the host analysis venv (Python 3.11) that ran every `tools/` script; SwissClim Evaluations pinned to a commit |
| `conda-tools.yml` | TempestExtremes, udunits, netCDF, ffmpeg, tectonic as in the Clariden miniforge base |

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r env/host-requirements.txt && uv pip install -e . --no-deps
conda env create -f env/conda-tools.yml          # only for Milton tracking, animations, LaTeX
```
