"""Model registry mapping our model names to earth2studio classes.

Each entry exposes:
  - the dotted import path of the e2s prognostic model class
  - the model time step in hours (used to translate `--lead-hours` to `--steps`)
  - whether the model is intrinsically probabilistic (stochastic per call)
  - which experiment role it plays here ('perturb_target' = deterministic
    model whose weights we perturb; 'baseline_prob' = probabilistic model
    we run with re-seeded samples to compare against)

The registry is the single point of truth used by `ai-ens infer --model NAME`,
the perturbation layer, and any future ensemble orchestration.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["perturb_target", "baseline_prob"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    e2s_class: str  # "earth2studio.models.px:GraphCastOperational"
    step_hours: int
    probabilistic: bool = False
    role: Role = "perturb_target"
    description: str = ""


REGISTRY: dict[str, ModelSpec] = {
    # Deterministic 0.25 deg / 6h - weights perturbed for ensemble construction.
    "graphcast_operational": ModelSpec(
        name="graphcast_operational",
        e2s_class="earth2studio.models.px:GraphCastOperational",
        step_hours=6,
        probabilistic=False,
        role="perturb_target",
        description="ECMWF-deployed GraphCast (0.25 deg, 6h step).",
    ),
    "sfno": ModelSpec(
        name="sfno",
        e2s_class="earth2studio.models.px:SFNO",
        step_hours=6,
        probabilistic=False,
        role="perturb_target",
        description="Spherical FNO (FourCastNet v2 lineage, 0.25 deg, 6h).",
    ),
    "aurora": ModelSpec(
        name="aurora",
        e2s_class="earth2studio.models.px:Aurora",
        step_hours=6,
        probabilistic=False,
        role="perturb_target",
        description="Microsoft Aurora (0.25 deg, 6h).",
    ),
    # Probabilistic 0.25 deg / 6h - re-seeded per member for ensemble.
    "fcn3": ModelSpec(
        name="fcn3",
        e2s_class="earth2studio.models.px:FCN3",
        step_hours=6,
        probabilistic=True,
        role="baseline_prob",
        description="FourCastNet v3 (probabilistic, 0.25 deg, 6h).",
    ),
    "atlas": ModelSpec(
        name="atlas",
        e2s_class="earth2studio.models.px:Atlas",
        step_hours=6,
        probabilistic=True,
        role="baseline_prob",
        description="NVIDIA Atlas-ERA5 (stochastic interpolant, 0.25 deg, 6h).",
    ),
    "aifsens": ModelSpec(
        name="aifsens",
        e2s_class="earth2studio.models.px:AIFSENS",
        step_hours=6,
        probabilistic=True,
        role="baseline_prob",
        description="ECMWF AIFS-ENS v1 (probabilistic, 0.25 deg, 6h).",
    ),
    # AIFS v1 deterministic, added as a 4th perturb target alongside aurora /
    # graphcast / sfno. Shares architecture with aifsens (only differs by
    # the conditional-layer-norm noise injection in the processor), so the
    # matched-architecture comparison "weight perturb of AIFS det" vs
    # "trained-stochastic AIFS-ENS" is the cleanest version of the paper's
    # central question. Same earth2studio[aifs] container as aifsens
    # (anemoi-models==0.5.1). AIFS v2 / AIFS2ENS skipped 2026-05-29 because
    # CDS lexicon in earth2studio is missing the wave-period-band heights
    # (h1012-h2530) that AIFS v2 requires -- see MEMORY.md.
    "aifs": ModelSpec(
        name="aifs",
        e2s_class="earth2studio.models.px:AIFS",
        step_hours=6,
        probabilistic=False,
        role="perturb_target",
        description="ECMWF AIFS Single v1 (deterministic, 0.25 deg, 6h, GNN+sliding-window transformer).",
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[name]


def import_class(spec: ModelSpec) -> Any:
    module_path, _, class_name = spec.e2s_class.partition(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if spec.name == "aurora":
        _patch_aurora_package(cls)
    return cls


def _patch_aurora_package(cls: Any) -> None:
    """Patch Aurora to use HuggingFace main branch instead of closed refs/pr/1.

    earth2studio hardcodes refs/pr/1 which was closed on HuggingFace (2025-06-23).
    The .npy static files only existed on that PR; main has .pickle format.
    """
    from earth2studio.models.auto.package import Package

    @classmethod  # type: ignore[misc]
    def patched_default_package(klass: type) -> Package:
        return Package(
            "hf://microsoft/aurora",
            cache_options={
                "cache_storage": Package.default_cache("aurora"),
                "same_names": True,
            },
        )

    @classmethod  # type: ignore[misc]
    def patched_load_model(klass: type, package: Package) -> Any:
        import pickle

        import numpy as np
        import torch
        from aurora import Aurora as AuroraModel

        # Static data always comes from the HF default package (small, cached).
        # The `package` arg may be a local perturbed copy with only the .ckpt.
        default_pkg = klass.load_default_package()
        static_path = default_pkg.resolve("aurora-0.25-static.pickle")
        with open(static_path, "rb") as f:
            static = pickle.load(f)
        z = torch.from_numpy(np.array(static["z"])[:-1])
        slt = torch.from_numpy(np.array(static["slt"])[:-1])
        lsm = torch.from_numpy(np.array(static["lsm"])[:-1])
        model = AuroraModel(use_lora=False)
        model.load_checkpoint_local(package.resolve("aurora-0.25-pretrained.ckpt"))
        model.eval()
        return klass(model, z, slt, lsm)

    cls.load_default_package = patched_default_package
    cls.load_model = patched_load_model


def load_model(name: str, package_root: str | None = None) -> tuple[Any, ModelSpec]:
    """Load an earth2studio prognostic model by registry name.

    If `package_root` is given, wrap it in `earth2studio.utils.package.Package`
    and use it instead of `load_default_package()` (so a user-perturbed
    checkpoint mirror can be loaded).
    """
    spec = get_spec(name)
    cls = import_class(spec)
    if package_root is None:
        package = cls.load_default_package()
    else:
        from earth2studio.models.auto.package import Package

        package = Package(package_root)
    model = cls.load_model(package)
    return model, spec


def steps_from_hours(spec: ModelSpec, lead_hours: int) -> int:
    if lead_hours % spec.step_hours != 0:
        raise ValueError(
            f"lead_hours={lead_hours} not divisible by {spec.name} step {spec.step_hours}h"
        )
    return lead_hours // spec.step_hours


__all__ = [
    "ModelSpec",
    "Role",
    "REGISTRY",
    "get_spec",
    "import_class",
    "load_model",
    "steps_from_hours",
]
