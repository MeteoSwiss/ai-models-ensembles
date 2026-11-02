---
name: feedback-eccodeslib-on-aarch64
description: "On CSCS aarch64 (Clariden), the pip `eccodes` wrapper alone does NOT load the libeccodes C library. Need to also install `eccodeslib` + `eckitlib` (newer eccodes >=2.39 split the C runtime into separate pip packages). `ecmwflibs` does NOT publish aarch64 wheels."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

When using the pip-installed `eccodes` Python wrapper on Clariden (aarch64 / GH200), `import eccodes` fails with:

```
RuntimeError: Cannot find the ecCodes library
```

Why: starting with eccodes-python >=2.39, the `libeccodes.so` / `libeckit.so` C runtime libraries are no longer bundled inside the `eccodes` wheel itself. They are split into separate pip packages: `eccodeslib` and `eckitlib`. `findlibs.find('eccodes')` returns None unless one of these is installed.

The older umbrella `ecmwflibs` package would normally cover this but it does NOT publish aarch64 Linux wheels (only x86_64). So on Clariden specifically we have to use the new split packages.

How to apply, for any project that uses earthkit-data, eccodes, cfgrib, or GRIB handling on Clariden:

```
# In requirements.txt, alongside eccodes:
eccodes==2.42.0       # the Python wrapper
eccodeslib>=2.47      # the C library (separate package since >=2.39)
eckitlib>=2.0         # dependency of eccodeslib
```

Or one-shot install if you're patching an existing venv:
```bash
uv pip install --no-cache eccodeslib eckitlib
```

Hit on 2026-05-28 setting up the IFS IC download repo; one-liner fix once identified. The `ecmwflibs` umbrella package is a tempting target but it has no aarch64 wheel and the fallback dependency tree is more brittle.

Verify it loads:
```bash
python -c "import eccodes; print('eccodes API:', eccodes.codes_get_api_version())"
```
