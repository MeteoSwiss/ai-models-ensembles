"""Canonical per-model colours, line styles and markers for every paper figure.

Single source of truth so a given model always renders in the same colour across
plots, table marker pills and captions. The LaTeX side mirrors these hex values in
figures/experiments_tables.tex (the \\definecolor block) - keep the two in sync.

Usage from a tools/ script:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/
    from model_colors import color_for, style_for, marker_for

From tools/milton/ (one level deeper):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""

from __future__ import annotations

# 8 headline baselines, canonical hex (mirrors experiments_tables.tex).
MODEL_COLORS = {
    "aurora_encoder": "#E67E22",  # orange
    "graphcast_all": "#27AE60",  # green
    "sfno_modes10": "#2980B9",  # blue
    "aifs_perturbed": "#8E44AD",  # purple
    "aifsens": "#8B5A2B",  # brown
    "atlas": "#C0392B",  # red
    "fcn3": "#D4A017",  # gold
    "ifs_ens": "#7F8C8D",  # grey
}

# Milton uses the weight+IC (_ic) variants of the post-hoc backbones; they are the
# same model, so they inherit the base colour (AIFS unified to purple everywhere).
_IC_ALIASES = {
    "aurora_encoder_ic": "aurora_encoder",
    "graphcast_all_ic": "graphcast_all",
    "sfno_modes10_ic": "sfno_modes10",
    "aifs_perturbed_ic": "aifs_perturbed",
}

# The ONLY place the AIFS weight+IC variant is drawn in a distinct colour: the
# weight-only vs weight+IC contrast figures (Milton F9 / Phase-5 SSR diagnostic),
# where the whole point is to tell the two AIFS runs apart on one axis.
AIFS_IC_CONTRAST = "#D81B60"  # pink

# Line style by role: post-hoc solid, trained-probabilistic dashed, classical dotted.
LINESTYLE = {
    "aurora_encoder": "-",
    "graphcast_all": "-",
    "sfno_modes10": "-",
    "aifs_perturbed": "-",
    "aifsens": "--",
    "atlas": "--",
    "fcn3": "--",
    "ifs_ens": ":",
}

# Marker by role: post-hoc circle, trained-probabilistic square, classical diamond.
MARKER = {
    "aurora_encoder": "o",
    "graphcast_all": "o",
    "sfno_modes10": "o",
    "aifs_perturbed": "o",
    "aifsens": "s",
    "atlas": "s",
    "fcn3": "s",
    "ifs_ens": "D",
}


def canon(name: str) -> str:
    """Map an _ic variant name to its canonical base model name."""
    return _IC_ALIASES.get(name, name)


def color_for(name: str) -> str:
    return MODEL_COLORS[canon(name)]


def style_for(name: str) -> str:
    return LINESTYLE[canon(name)]


def marker_for(name: str) -> str:
    return MARKER[canon(name)]
