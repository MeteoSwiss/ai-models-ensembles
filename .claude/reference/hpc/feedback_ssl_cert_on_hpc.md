---
name: feedback-ssl-cert-on-hpc
description: "ALWAYS export SSL_CERT_FILE + REQUESTS_CA_BUNDLE to certifi's bundle when a Python job on a CSCS compute node makes HTTPS calls (ECMWF API, HuggingFace, GitHub, etc). Compute nodes do NOT ship a system CA bundle."
metadata:
  node_type: memory
  type: feedback
  originSessionId: e19351ee-7832-4e68-9914-6f5c6c57c11c
---

CSCS Clariden compute nodes do not have `/etc/ssl/certs/ca-certificates.crt` or `/etc/pki/tls/certs/ca-bundle.crt` -- the standard system CA bundle paths. Any Python code that opens an HTTPS connection from a compute node fails certificate verification:

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
  certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
```

This bites every Python HTTPS-using library on the cluster -- saw it 2026-05-28 with `ecmwf-api-client` for MARS access, and it's the same root cause that would hit `huggingface_hub`, `requests`, `urllib.request`, `aiohttp`, etc.

Why: login nodes have the system bundle; compute nodes are stripped down. The compute-node Python's `ssl` module falls back to OS defaults which don't exist.

How to apply, for every sbatch wrap that runs Python with HTTPS:
```bash
# certifi ships its own bundle with every venv that depends on it (most do).
CERT=$(.venv/bin/python -c 'import certifi; print(certifi.where())')
sbatch ... --wrap="
    export SSL_CERT_FILE=$CERT && \
    export REQUESTS_CA_BUNDLE=$CERT && \
    .venv/bin/python my_script.py"
```

For libraries that use `urllib` directly (`ecmwfapi`): `SSL_CERT_FILE` is what they read.
For libraries built on `requests`: `REQUESTS_CA_BUNDLE` is what they read.
Setting both is cheap insurance.

Make sure `certifi` is in the venv's `requirements.txt`. It usually is implicit (many packages depend on it) but explicit is safer.

System-wide rule, same flavour as [[feedback-cluster-max-resources]] and [[feedback-never-use-tmp-on-hpc]]. Take it to every CSCS HPC project that uses Python HTTPS from a compute node.

Bonus: also `export PYTHONUNBUFFERED=1` and run `python -u` so stdout flushes line-by-line. Otherwise SLURM log files stay empty for hours while Python block-buffers prints to a redirected file, and you can't tell whether a long blocking call (MARS, HuggingFace) is making progress.
