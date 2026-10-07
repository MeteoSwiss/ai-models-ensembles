"""Fresh-per-step weight perturbation for GraphCast all-weights (Phase 6c).

GraphCast analog of [[phase6-fresh-per-step-weight]] for the production
``graphcast_all`` cell: every learned parameter gets the same multiplicative
noise as the frozen ``_perturb_npz`` path, ``w * (1 + sigma * N(0,1))``, but
the noise is re-sampled every ``refresh_every`` AR steps (1 = fresh per step;
T = frozen per member).

earth2studio's ``GraphCastOperational.run_forward`` is
``drop_state(partial(jit(apply), params=ckpt.params, state={}))``, so params
enter the jitted function as an argument. Overriding ``params=`` per call
with a same-shape pytree therefore does not trigger a recompile. The
model_config / task_config live outside ``ckpt.params`` and are never touched.

Gated by env vars when running inside e2s_inference._model_for_member:
    GRAPHCAST_FRESH=1
    GRAPHCAST_FRESH_SIGMA=<float>          # e.g. 0.01
    GRAPHCAST_FRESH_REFRESH_EVERY=<int>    # default 1
"""

from __future__ import annotations

import os

import numpy as np


def _mix_seed(base_seed: int, unit_idx: int, step: int) -> int:
    """Avalanche-mix (member, tensor, epoch) into an independent 63-bit seed.

    Same splitmix64 finaliser as the Aurora/AIFS/SFNO fresh hooks.
    """
    z = (
        int(base_seed) * 0x9E3779B97F4A7C15
        + int(unit_idx) * 0xD1B54A32D192ED03
        + int(step) * 0xF1357AEA2E62A9C5
    ) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0x7FFFFFFFFFFFFFFF


def install_fresh_weight_noise(
    model,
    sigma: float,
    base_seed: int,
    refresh_every: int = 1,
) -> None:
    """Wrap ``model.run_forward`` so each call sees the current epoch's noisy params."""
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(model.ckpt.params)
    leaves = [np.asarray(w) for w in leaves]
    n_float = sum(1 for w in leaves if np.issubdtype(w.dtype, np.inexact))
    if n_float == 0:
        raise RuntimeError("GraphCast fresh-perturbation: ckpt.params has no float leaves.")
    orig_run_forward = model.run_forward
    state = {"step": 0, "epoch": None, "params": None}

    def run_forward(**kw):
        state["step"] += 1
        epoch = ((state["step"] - 1) // refresh_every) + 1
        if epoch != state["epoch"]:
            noisy = []
            for unit_idx, w in enumerate(leaves):
                if not np.issubdtype(w.dtype, np.inexact):
                    noisy.append(w)
                    continue
                rng = np.random.default_rng(_mix_seed(base_seed, unit_idx, epoch))
                noise = rng.standard_normal(size=w.shape, dtype=np.float32).astype(w.dtype)
                noisy.append((w * (1.0 + sigma * noise)).astype(w.dtype, copy=False))
            state["params"] = jax.tree_util.tree_unflatten(treedef, noisy)
            state["epoch"] = epoch
            print(
                f"[GRAPHCAST_FRESH] resampled step={state['step']} epoch={epoch} "
                f"refresh_every={refresh_every} sigma={sigma:.4g} n_float={n_float} "
                f"|delta|/|w| (first)={float(np.abs(noisy[0] - leaves[0]).mean() / (np.abs(leaves[0]).mean() + 1e-12)):.4g}",
                flush=True,
            )
        return orig_run_forward(params=state["params"], **kw)

    model.run_forward = run_forward
    print(
        f"[GRAPHCAST_FRESH] wrapped run_forward "
        f"(sigma={sigma}, refresh_every={refresh_every}, "
        f"base_seed={base_seed}, n_leaves={len(leaves)}, n_float={n_float}, "
        f"n_params={sum(w.size for w in leaves)})",
        flush=True,
    )


def maybe_install_from_env(model, base_seed: int) -> None:
    """Check env vars and wrap run_forward if GRAPHCAST_FRESH=1."""
    if os.environ.get("GRAPHCAST_FRESH", "0") != "1":
        return
    sigma = float(os.environ.get("GRAPHCAST_FRESH_SIGMA", "0.0"))
    refresh_every = int(os.environ.get("GRAPHCAST_FRESH_REFRESH_EVERY", "1"))
    if sigma <= 0:
        raise ValueError("GRAPHCAST_FRESH=1 but GRAPHCAST_FRESH_SIGMA is unset/invalid.")
    if refresh_every < 1:
        raise ValueError(f"GRAPHCAST_FRESH_REFRESH_EVERY={refresh_every} must be >= 1.")
    install_fresh_weight_noise(model, sigma, base_seed, refresh_every=refresh_every)
