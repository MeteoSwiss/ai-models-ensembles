"""Fresh-per-step weight perturbation for Aurora encoder (Phase 6c).

Aurora analog of [[phase6-fresh-per-step-weight]]: at every forward call to
the root encoder module, save the current values of every parameter in
``net.encoder.*``, apply multiplicative noise ``w *= (1 + sigma * N(0,1))``
in-place, run the forward pass, then restore the saved values in the
post-hook. ``refresh_every`` controls how often the noise tensor is
re-sampled along the rollout (1 = fresh per step; T = frozen per member).

Gated by env vars when running inside e2s_inference._model_for_member:
    AURORA_FRESH=1
    AURORA_FRESH_SIGMA=<float>             # e.g. 0.025 or 0.035
    AURORA_FRESH_REFRESH_EVERY=<int>       # default 1
"""

from __future__ import annotations

import os

import torch


def _mix_seed(base_seed: int, unit_idx: int, step: int) -> int:
    """Avalanche-mix (member, tensor, epoch) into an independent 63-bit seed.

    base_seed carries the member (seed + member_id); unit_idx separates the
    per-tensor draws; step is the refresh epoch. A plain ``seed + step`` made
    member m at epoch e collide with member m+1 at epoch e-1, and reused one
    seed across every tensor in a step; the splitmix64 finaliser with distinct
    odd multipliers per axis removes both.
    """
    z = (
        int(base_seed) * 0x9E3779B97F4A7C15
        + int(unit_idx) * 0xD1B54A32D192ED03
        + int(step) * 0xF1357AEA2E62A9C5
    ) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


def _seeded_noise_like(
    p: torch.Tensor,
    seed: int,
    unit_idx: int,
    step: int,
) -> torch.Tensor:
    """Draw a noise tensor matching ``p`` deterministically per (seed, tensor, step).

    Step indexing collapses every ``refresh_every`` AR steps into one noise
    epoch upstream, so this function only sees the per-epoch index.
    """
    gen = torch.Generator(device=p.device)
    gen.manual_seed(_mix_seed(seed, unit_idx, step))
    return torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)


def _find_encoder(root: torch.nn.Module) -> torch.nn.Module:
    """Walk the module tree and locate Aurora's encoder submodule.

    earth2studio wraps the underlying Aurora model behind a thin adapter; the
    state-dict key prefix is `net.encoder.*` but the live module path may be
    `.model.net.encoder` or similar. Walk named_modules() and return the first
    submodule whose name ENDS in `net.encoder` (or `encoder` as a fallback)
    AND has roughly the documented 33 learnable parameters.
    """
    candidates_strict = []
    candidates_loose = []
    for name, sub in root.named_modules():
        if not isinstance(sub, torch.nn.Module):
            continue
        n_params = sum(1 for p in sub.parameters() if p.requires_grad)
        if name.endswith("net.encoder") or name.endswith(".net.encoder"):
            candidates_strict.append((name, sub, n_params))
        elif name.endswith(".encoder") or name == "encoder":
            candidates_loose.append((name, sub, n_params))
    pool = candidates_strict or candidates_loose
    if not pool:
        head = [n for n, _ in root.named_children()][:20]
        raise RuntimeError(
            "Aurora fresh-perturbation: no encoder submodule found via "
            "`net.encoder` or `.encoder` suffix. Top-level children: "
            f"{head}"
        )
    # Prefer the candidate whose direct learnable-tensor count is closest to
    # the documented 33-tensor encoder, falling back to first match.
    pool.sort(key=lambda t: abs(t[2] - 33))
    name, sub, n_params = pool[0]
    print(
        f"[AURORA_FRESH] selected encoder module '{name}' "
        f"({n_params} learnable params, target ~33)",
        flush=True,
    )
    return sub


def install_fresh_encoder_hooks(
    model: torch.nn.Module,
    sigma: float,
    base_seed: int,
    refresh_every: int = 1,
) -> list:
    """Install pre/post hooks on the discovered Aurora encoder submodule.

    Returns the list of hook handles so the caller can uninstall them.
    Every learnable parameter under the discovered encoder subtree gets
    multiplicative noise.
    """
    encoder = _find_encoder(model)
    params = [(name, p) for name, p in encoder.named_parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("Aurora fresh-perturbation: encoder has no learnable parameters.")

    saved: dict[int, torch.Tensor] = {}
    step_counter = {"i": 0}

    def pre_hook(module, inputs):
        step_counter["i"] += 1
        step = step_counter["i"]
        noise_step = ((step - 1) // refresh_every) + 1
        saved.clear()
        for unit_idx, (name, p) in enumerate(params):
            saved[id(p)] = p.data.clone()
            noise = _seeded_noise_like(p, base_seed, unit_idx, noise_step)
            p.data.mul_(1.0 + sigma * noise)
        if step == 1:
            sample = params[0][1]
            print(
                f"[AURORA_FRESH] PRE_HOOK fired step={step} "
                f"refresh_every={refresh_every} sigma={sigma:.4g} "
                f"n_params={len(params)} "
                f"|delta|mean(first)={sigma * float(_seeded_noise_like(sample, base_seed, 0, noise_step).abs().mean()):.4g}",
                flush=True,
            )

    def post_hook(module, inputs, outputs):
        for _name, p in params:
            p.data.copy_(saved[id(p)])
        saved.clear()

    handles = [
        encoder.register_forward_pre_hook(pre_hook),
        encoder.register_forward_hook(post_hook),
    ]
    print(
        f"[AURORA_FRESH] installed encoder hooks "
        f"(sigma={sigma}, refresh_every={refresh_every}, "
        f"base_seed={base_seed}, n_params={len(params)})",
        flush=True,
    )
    return handles


def maybe_install_from_env(model, base_seed: int) -> list:
    """Check env vars and install hooks if AURORA_FRESH=1."""
    if os.environ.get("AURORA_FRESH", "0") != "1":
        return []
    sigma = float(os.environ.get("AURORA_FRESH_SIGMA", "0.0"))
    refresh_every = int(os.environ.get("AURORA_FRESH_REFRESH_EVERY", "1"))
    if sigma <= 0:
        raise ValueError("AURORA_FRESH=1 but AURORA_FRESH_SIGMA is unset/invalid.")
    if refresh_every < 1:
        raise ValueError(f"AURORA_FRESH_REFRESH_EVERY={refresh_every} must be >= 1.")
    return install_fresh_encoder_hooks(model, sigma, base_seed, refresh_every=refresh_every)
