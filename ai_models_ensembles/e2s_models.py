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
}


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[name]


def import_class(spec: ModelSpec) -> Any:
    module_path, _, class_name = spec.e2s_class.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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
        from earth2studio.utils.package import Package

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
