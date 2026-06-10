"""Fresh-per-step weight perturbation for AIFS decoder (Phase 6c).

AIFS analog of [[phase6-fresh-per-step-weight]]: at every forward call to
the decoder root module, save the current values of every parameter in
``model.decoder.*``, apply multiplicative noise ``w *= (1 + sigma * N(0,1))``
in-place, run the forward pass, then restore the saved values in the
post-hook. ``refresh_every`` controls how often the noise tensor is
re-sampled along the rollout (1 = fresh per step; T = frozen per member).

The AIFS model graph from earth2studio is an AnemoiModelInterface wrapping a
processor module also named ``model``, so the decoder is at the path
``<root>.model.decoder``. ``find_decoder`` walks the module tree once to
locate it; we use the deepest common ancestor for the pre/post hook.

Gated by env vars when running inside e2s_inference._model_for_member:
    AIFS_FRESH=1
    AIFS_FRESH_SIGMA=<float>             # e.g. 0.0275 or 0.0385
    AIFS_FRESH_REFRESH_EVERY=<int>       # default 1
"""

from __future__ import annotations

import os

import torch


def _seeded_noise_like(p: torch.Tensor, seed: int, step: int) -> torch.Tensor:
    gen = torch.Generator(device=p.device)
    gen.manual_seed(int(seed + step) & 0x7FFFFFFF)
    return torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)


def _find_decoder(root: torch.nn.Module) -> torch.nn.Module:
    """Walk the module tree and return the first module ending in `.decoder`.

    Looks under common AIFS wrappers (.model, .module) before falling back to
    a name-based scan over `named_modules()`. Raises if nothing matches.
    """
    for path in ("model.decoder", "module.decoder", "decoder"):
        cur: torch.nn.Module | None = root
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, torch.nn.Module):
                return cur
        except AttributeError:
            continue
    for name, sub in root.named_modules():
        if name.endswith(".decoder") and isinstance(sub, torch.nn.Module):
            return sub
    raise RuntimeError(
        "AIFS fresh-perturbation: could not locate a `.decoder` submodule. "
        f"Top-level attributes: {[n for n, _ in root.named_children()][:20]}"
    )


def install_fresh_decoder_hooks(
    model: torch.nn.Module,
    sigma: float,
    base_seed: int,
    refresh_every: int = 1,
) -> list:
    """Install pre/post hooks on the AIFS decoder root.

    Returns the list of hook handles so the caller can uninstall them.
    Every learnable parameter under ``decoder.*`` gets multiplicative noise.
    """
    decoder = _find_decoder(model)
    params = [(name, p) for name, p in decoder.named_parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("AIFS fresh-perturbation: decoder has no learnable parameters.")

    saved: dict[int, torch.Tensor] = {}
    step_counter = {"i": 0}

    def pre_hook(module, inputs):
        step_counter["i"] += 1
        step = step_counter["i"]
        noise_step = ((step - 1) // refresh_every) + 1
        saved.clear()
        for _name, p in params:
            saved[id(p)] = p.data.clone()
            noise = _seeded_noise_like(p, base_seed, noise_step)
            p.data.mul_(1.0 + sigma * noise)
        if step == 1:
            sample = params[0][1]
            print(
                f"[AIFS_FRESH] PRE_HOOK fired step={step} "
                f"refresh_every={refresh_every} sigma={sigma:.4g} "
                f"n_params={len(params)} "
                f"|delta|mean(first)={sigma * float(_seeded_noise_like(sample, base_seed, noise_step).abs().mean()):.4g}",
                flush=True,
            )

    def post_hook(module, inputs, outputs):
        for _name, p in params:
            p.data.copy_(saved[id(p)])
        saved.clear()

    handles = [
        decoder.register_forward_pre_hook(pre_hook),
        decoder.register_forward_hook(post_hook),
    ]
    print(
        f"[AIFS_FRESH] installed decoder hooks "
        f"(sigma={sigma}, refresh_every={refresh_every}, "
        f"base_seed={base_seed}, n_params={len(params)})",
        flush=True,
    )
    return handles


def maybe_install_from_env(model, base_seed: int) -> list:
    """Check env vars and install hooks if AIFS_FRESH=1."""
    if os.environ.get("AIFS_FRESH", "0") != "1":
        return []
    sigma = float(os.environ.get("AIFS_FRESH_SIGMA", "0.0"))
    refresh_every = int(os.environ.get("AIFS_FRESH_REFRESH_EVERY", "1"))
    if sigma <= 0:
        raise ValueError("AIFS_FRESH=1 but AIFS_FRESH_SIGMA is unset/invalid.")
    if refresh_every < 1:
        raise ValueError(f"AIFS_FRESH_REFRESH_EVERY={refresh_every} must be >= 1.")
    return install_fresh_decoder_hooks(model, sigma, base_seed, refresh_every=refresh_every)
