#!/usr/bin/env python3
import os
from typing import List

out: List[str] = []


def add(line: str):
    out.append(line)


# First, import JAX to preload its CUDA/cuDNN from pip wheels and report status
try:
    from importlib import metadata as md

    import jax
    from packaging.requirements import Requirement

    dev_kinds = []
    for d in jax.devices():
        kind = getattr(d, "device_kind", getattr(d, "platform", "unknown"))
        dev_kinds.append(str(kind))
    add("jax devices: " + ", ".join(dev_kinds))
    add(f"jax backend: {jax.default_backend()}")

    # If CUDA wheels are present, check nvidia-cudnn-cu12 against jaxlib requirements
    try:
        cudnn_ver = md.version("nvidia-cudnn-cu12")
        add(f"nvidia-cudnn-cu12: {cudnn_ver}")
        jaxlib_req = None
        for r in md.requires("jaxlib") or []:
            req = Requirement(r)
            if req.name == "nvidia-cudnn-cu12":
                jaxlib_req = str(req.specifier)
                break
        if jaxlib_req:
            add(f"jaxlib requires nvidia-cudnn-cu12{jaxlib_req}")
        else:
            add("jaxlib cudnn requirement not found in metadata")
    except md.PackageNotFoundError:
        add("nvidia-cudnn-cu12: not installed (CPU JAX or local CUDA)")
except Exception as e:
    add(f"jax error: {e}")

# Torch info (after JAX)
try:
    import torch

    add(
        f"torch {torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}"
    )
    if torch.cuda.is_available():
        add(f"torch device 0: {torch.cuda.get_device_name(0)}")
        try:
            add(f"torch cudnn version: {torch.backends.cudnn.version()}")
        except Exception:
            pass
except Exception as e:
    add(f"torch error: {e}")

# Helpful env info
ld = os.environ.get("LD_LIBRARY_PATH")
if ld:
    add(f"LD_LIBRARY_PATH: {ld}")

print("\n".join(out))
