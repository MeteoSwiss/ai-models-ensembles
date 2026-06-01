"""Fresh-per-step weight perturbation for SFNO modes10 (Phase 6a-prime).

For each forward call to a spectral conv module (named ``*.filter.filter``),
re-sample complex64 Gaussian noise of shape ``weight[..., :mode_cut]`` and
apply a multiplicative perturbation ``weight *= (1 + sigma * noise)`` for
the duration of that forward call, then restore the original weight in the
post-forward hook.

This is the WEIGHT-domain analogue of the temporal-budget rule:
    sigma_fresh = sigma_persistent * sqrt(T)
so the cumulative variance over T rollout steps matches a single-shot
checkpoint perturbation at sigma_persistent. See
[[phase6-fresh-per-step-weight]] for the math and [[sigma-sweep-design]]
for the variance-budget framing.

Gated by env vars when running inside e2s_inference._model_for_member:
    SFNO_FRESH=1
    SFNO_FRESH_SIGMA=<float>          # e.g. 0.60 or 1.93
    SFNO_FRESH_MODE_CUT=<int>         # e.g. 10
"""

from __future__ import annotations

import math
import os

import torch


def _seeded_complex_noise(
    weight: torch.Tensor,
    mode_cut: int,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Draw fresh complex64 noise of shape weight[..., :mode_cut].

    Each (member, step) pair gets a unique RNG seed derived from
    ``seed + step``, so noise is reproducible per (member, step) and
    completely decorrelated across steps within a member -- the
    random-walk property the spatial-mean-spread theory needs.
    """
    gen = torch.Generator(device=weight.device)
    gen.manual_seed(int(seed + step))
    shape = (*weight.shape[:-1], mode_cut)
    real = torch.randn(*shape, generator=gen, device=weight.device, dtype=torch.float32)
    imag = torch.randn(*shape, generator=gen, device=weight.device, dtype=torch.float32)
    return torch.complex(real, imag) / math.sqrt(2.0)


def install_fresh_modes_hooks(
    model: torch.nn.Module,
    sigma: float,
    mode_cut: int,
    base_seed: int,
) -> list:
    """Install per-call fresh-noise hooks on every SFNO spectral conv module.

    Returns the list of hook handles so the caller can uninstall them.

    The hooks operate per (member, step):
      pre_hook:  save the original weight, sample fresh noise, write the
                 perturbed weight in-place into module.weight.data.
      post_hook: restore the original weight from the saved copy.

    Step indexing is per-module: each module owns its own counter, since
    SFNO's rollout calls each spectral conv block once per AR step, so a
    module-local counter is well-defined and trivially per-AR-step.
    """
    saved: dict[int, torch.Tensor] = {}
    step_per_module: dict[int, int] = {}
    handles: list = []

    def _make_pre(weight: torch.Tensor, mod_name: str):
        mod_id = id(weight)
        step_per_module[mod_id] = 0

        def pre_hook(module, inputs):
            step_per_module[mod_id] += 1
            step = step_per_module[mod_id]
            saved[mod_id] = weight.data.clone()
            noise = _seeded_complex_noise(weight, mode_cut, base_seed, step)
            pre_slice = weight.data[..., :mode_cut].clone()
            weight.data[..., :mode_cut] = weight.data[..., :mode_cut] * (
                1.0 + sigma * noise.to(weight.dtype)
            )
            if step == 1:
                delta = (weight.data[..., :mode_cut] - pre_slice).abs().mean().item()
                print(
                    f"[SFNO_FRESH] PRE_HOOK fired {mod_name} step={step} "
                    f"|noise|mean={noise.abs().mean().item():.4g} "
                    f"|delta_weight|mean={delta:.6g}",
                    flush=True,
                )

        return pre_hook

    def _make_post(weight: torch.Tensor):
        mod_id = id(weight)

        def post_hook(module, inputs, outputs):
            weight.data.copy_(saved.pop(mod_id))

        return post_hook

    matched = []
    for name, module in model.named_modules():
        if not name.endswith(".filter.filter"):
            continue
        if not hasattr(module, "weight"):
            continue
        w = module.weight
        if w.shape[-1] < mode_cut:
            raise ValueError(f"{name}.weight last axis = {w.shape[-1]} < mode_cut = {mode_cut}")
        matched.append((name, tuple(w.shape), w.dtype))
        handles.append(module.register_forward_pre_hook(_make_pre(w, name)))
        handles.append(module.register_forward_hook(_make_post(w)))

    if not handles:
        # Diagnostic: list every module name so we can see what the SFNO graph
        # actually exposes when the conventional `.filter.filter` lookup fails.
        all_names = [n for n, _ in model.named_modules()]
        sample = [n for n in all_names if "filter" in n.lower()][:30]
        raise RuntimeError(
            "SFNO fresh-perturbation hooks not installed: no `*.filter.filter` "
            f"modules with a `.weight` attribute found. Total modules in tree: "
            f"{len(all_names)}. Names containing 'filter' (first 30): {sample}"
        )

    print(
        f"[SFNO_FRESH] installed {len(matched)} hooks "
        f"(sigma={sigma}, mode_cut={mode_cut}, base_seed={base_seed}):",
        flush=True,
    )
    for n, s, d in matched:
        print(f"[SFNO_FRESH]   {n}  shape={s}  dtype={d}", flush=True)
    return handles


def maybe_install_from_env(model, base_seed: int) -> list:
    """Check env vars and install hooks if SFNO_FRESH=1.

    Returns the handle list (empty if not enabled).
    """
    if os.environ.get("SFNO_FRESH", "0") != "1":
        return []
    sigma = float(os.environ.get("SFNO_FRESH_SIGMA", "0.0"))
    mode_cut = int(os.environ.get("SFNO_FRESH_MODE_CUT", "0"))
    if sigma <= 0 or mode_cut <= 0:
        raise ValueError(
            "SFNO_FRESH=1 but SFNO_FRESH_SIGMA or SFNO_FRESH_MODE_CUT is unset/invalid."
        )
    return install_fresh_modes_hooks(model, sigma, mode_cut, base_seed)
