#!/usr/bin/env python3
"""Schematic of weight-perturbation scheme across the 3 ablation models.

Output: figures/perturbation_schematic.{svg,pdf}

The SVG is fully editable in Inkscape / Illustrator / Affinity Designer.
Numbers and sigmas come from the verified 2026-05-18 checkpoint dumps
(see memory/checkpoint_perturbation_audit.md) and the Phase 2 scaling
rule sigma_partial = sigma_full * sqrt(N_total / N_partial) at
sigma_full = 0.01.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# -- Style --------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-compatible serif math, matches the AMS LaTeX paper
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",  # keep text as text in the SVG
    }
)

# Colour palette (mute, paper-friendly)
COL_ENC = "#E08B3A"  # warm orange
COL_PROC = "#2C7A8C"  # deep teal
COL_DEC = "#7BB369"  # sage green
COL_AUX = "#A0A0A0"  # grey for residual/aux
COL_NOISE = "#C2493D"  # noise overlay accent
COL_TEXT = "#1F2A36"
COL_MUTED = "#5C6B7A"
COL_GRID = "#E9ECEF"

SIGMA_FULL = 0.01


def sigma_scaled(n_total: int, n_partial: int) -> float:
    """sigma_partial = sigma_full * sqrt(N_total / N_partial)."""
    return SIGMA_FULL * math.sqrt(n_total / n_partial)


# -- Model spec ---------------------------------------------------------------

MODELS = [
    {
        "name": "Aurora",
        "subtitle": "Swin-Transformer 3D U-Net",
        "n_total": 644,
        "fmt": "float32 only",
        "groups": [
            {
                "key": "encoder",
                "n": 33,
                "kind": "encoder",
                "detail": "Perceiver-IO\n14 embed | 6 norm | 7 attn | 6 MLP",
                "shape": "(weights of perceiver and embeddings)",
            },
            {
                "key": "backbone",
                "n": 594,
                "kind": "processor",
                "detail": "48 Swin-3D blocks\nattn (192) + MLP (192) +\nLayerNorm (192) + AdaLN mod (192)",
                "shape": "Swin attention windows over 3D tokens",
            },
            {
                "key": "decoder",
                "n": 17,
                "kind": "decoder",
                "detail": "Perceiver-IO output\n2 embed | 4 norm | 3 attn | 4 MLP | 4 head",
                "shape": "(weights of perceiver and output heads)",
            },
        ],
    },
    {
        "name": "SFNO",
        "subtitle": "Spherical Fourier Neural Operator",
        "n_total": 87,
        "fmt": "79 float32 + 8 complex64",
        "groups": [
            {
                "key": "encoder",
                "n": 3,
                "kind": "encoder",
                "detail": "Conv2d input proj\n(channels in -> embed)",
                "shape": "module.model.encoder.fwd.*",
            },
            {
                "key": "processor",
                "n": 80,
                "kind": "processor",
                "detail": "8 SFNO blocks x 10 tensors each\n"
                "1 complex64 spectral conv (384, 384, 240)\n"
                "+ 4 MLP + 4 norms + 1 skip",
                "shape": "module.model.blocks.0..7",
            },
            {
                "key": "decoder",
                "n": 3,
                "kind": "decoder",
                "detail": "Conv2d output proj\n(embed -> channels out)",
                "shape": "module.model.decoder.fwd.*",
            },
            {
                "key": "residual",
                "n": 1,
                "kind": "aux",
                "detail": "Linear skip\n(input -> output)",
                "shape": "residual_transform.weight",
            },
        ],
    },
    {
        "name": "GraphCast",
        "subtitle": "Multi-Mesh Graph Neural Network",
        "n_total": 264,
        "fmt": "float32 .npz (3 model_config scalars skipped)",
        "groups": [
            {
                "key": "g2m",
                "n": 36,
                "kind": "encoder",
                "detail": "grid -> mesh GNN\n(encoder edge + node MLPs)",
                "shape": "params:grid2mesh_gnn.*",
            },
            {
                "key": "m2m",
                "n": 198,
                "kind": "processor",
                "detail": "16 message-passing rounds\non multi-mesh (lvl 0..6)\n~12 tensors per round",
                "shape": "params:mesh_gnn.*",
            },
            {
                "key": "m2g",
                "n": 30,
                "kind": "decoder",
                "detail": "mesh -> grid GNN\n(decoder edge + node MLPs)",
                "shape": "params:mesh2grid_gnn.*",
            },
        ],
    },
    {
        "name": "AIFS",
        "subtitle": "GNN + latitude-band transformer",
        "n_total": 242,
        "fmt": "float32 anemoi .ckpt (8 normaliser stats skipped)",
        "groups": [
            {
                "key": "encoder",
                "n": 34,
                "kind": "encoder",
                "detail": "grid -> O96 mesh GNN\n+ node-attribute embeddings\n(emb_nodes, emb_edges, proc)",
                "shape": "model.encoder.* + model.node_attributes.*",
            },
            {
                "key": "processor",
                "n": 176,
                "kind": "processor",
                "detail": "16 sliding-window transformer\nblocks (attn + MLP + 2 norms)\non O96 mesh, latitude bands",
                "shape": "model.processor.proc.*",
            },
            {
                "key": "decoder",
                "n": 32,
                "kind": "decoder",
                "detail": "O96 mesh -> grid GNN\n+ node_data_extractor head",
                "shape": "model.decoder.*",
            },
        ],
    },
]


def kind_color(kind: str) -> str:
    return {"encoder": COL_ENC, "processor": COL_PROC, "decoder": COL_DEC, "aux": COL_AUX}[kind]


# -- Helpers ------------------------------------------------------------------


def gaussian_path(cx, cy, w, h, n_pts=60, sigma=0.3):
    """Return an Nx2 Path approximating a Gaussian curve in the box (cx,cy,w,h)."""
    x = np.linspace(-3, 3, n_pts)
    y = np.exp(-0.5 * (x / 1.0) ** 2)
    x_norm = (x + 3) / 6  # [0,1]
    y_norm = y / y.max()  # [0,1]
    xs = cx - w / 2 + x_norm * w
    ys = cy - h / 2 + y_norm * h
    verts = list(zip(xs, ys))
    return verts


def draw_gaussian_glyph(ax, cx, cy, w=0.22, h=0.10, colour=COL_NOISE):
    """Tiny inline Gaussian curve to denote noise injection."""
    pts = gaussian_path(cx, cy, w, h)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=colour, lw=0.9, alpha=0.85, zorder=5)
    # Centre tick
    ax.plot([cx, cx], [cy - h / 2, cy - h / 2 + 0.02], color=colour, lw=0.7, alpha=0.7)


def draw_box(ax, x, y, w, h, colour, label, n, sigma, detail, highlight=None):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.022",
        facecolor=colour,
        edgecolor=colour,
        linewidth=1.0,
        alpha=0.92,
        zorder=2,
    )
    ax.add_patch(box)
    # Inner soft highlight
    overlay = FancyBboxPatch(
        (x + 0.006, y + 0.006),
        w - 0.012,
        h - 0.012,
        boxstyle="round,pad=0,rounding_size=0.018",
        facecolor="white",
        edgecolor="none",
        alpha=0.18,
        zorder=3,
    )
    ax.add_patch(overlay)

    # Inside the box, top-to-bottom:
    #   title (label) / N = ... / sigma = ... / detail (italic)
    ax.text(
        x + w / 2,
        y + h * 0.84,
        label,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h * 0.65,
        rf"$N_\mathrm{{group}} = {n}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="white",
        alpha=0.97,
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h * 0.49,
        rf"$\sigma_\mathrm{{group}} = {sigma:.4f}$",
        ha="center",
        va="center",
        fontsize=12.5,
        color="white",
        alpha=0.97,
        zorder=5,
    )

    # Detail (italic, multi-line) at the bottom of the box
    ax.text(
        x + w / 2,
        y + h * 0.06,
        detail,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="white",
        alpha=0.92,
        style="italic",
        zorder=5,
        linespacing=1.2,
    )


def draw_arrow(ax, x1, x2, y, colour=COL_MUTED):
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="-|>",
        mutation_scale=12,
        color=colour,
        lw=1.4,
        zorder=4,
    )
    ax.add_patch(arrow)


# -- Layout -------------------------------------------------------------------


# -- Layout constants ---------------------------------------------------------
# Normalized axis coords [0, 1] x [0, 1]; the figure is 13 x 9 inches.
LEFT_COL_X = 0.02  # left edge of model-label column
LEFT_COL_W = 0.21  # width of label column (fits longest subtitle)
BOX_AREA_X = LEFT_COL_X + LEFT_COL_W + 0.015  # left edge of box area
BOX_AREA_W = 1.0 - BOX_AREA_X - 0.02
# Row layout scales with len(MODELS). 3 models = original geometry; 4+
# compresses ROW_HEIGHT and the label-block offsets proportionally to keep
# inter-row gaps positive and avoid overlap with title (top edge ~0.94) and
# legend (bottom edge ~0.03).
_n_models = len(MODELS)
if _n_models == 3:
    ROW_HEIGHT = 0.23
    ROW_CENTERS = [0.75, 0.46, 0.17]
    _LABEL_SCALE = 1.0
else:
    # Available band is [_bot, _top]; rows are evenly spaced inside it.
    # The band is kept compact and centred so the top row clears the title
    # and the bottom row clears the footer legend (the boxes had been
    # overlapping both). Boxes take a large share of the pitch so the rows
    # sit close together.
    _top, _bot = 0.82, 0.15
    _span = _top - _bot
    _pitch = _span / (_n_models - 1)
    ROW_CENTERS = [_top - _pitch * i for i in range(_n_models)]
    # Boxes occupy ~80% of the row pitch -> ~20% inter-row gap. Label-column
    # offsets scale to the row pitch so icon, name, subtitle, N_total fit.
    ROW_HEIGHT = 0.80 * _pitch
    _LABEL_SCALE = min(1.0, _pitch / 0.28)  # original 3-row pitch was 0.28


def width_for_n(n: int, n_max: int, w_min=0.12, w_max=0.27) -> float:
    """Log-scale box width by tensor count, bounded."""
    if n <= 1:
        return w_min
    f = math.log10(n) / math.log10(max(n_max, 10))
    return w_min + (w_max - w_min) * f


# -- Iconographic mini-schematics --------------------------------------------


def draw_icon_unet(ax, x, y, w, h):
    """Aurora -- U-Net: descending then ascending rectangle hierarchy.

    Drawn as a row of seven small filled rectangles whose heights form a
    symmetric 'V': tall -> short -> tall, with a narrow bottleneck.
    """
    n = 7
    pad = 0.005
    gap = (w - 2 * pad) / (n + (n - 1) * 0.25)  # rect width
    spacing = gap * 1.25
    cx = x + pad
    # symmetric heights: 1.0, 0.75, 0.5, 0.3, 0.5, 0.75, 1.0 (in units of h)
    hs = [1.0, 0.75, 0.5, 0.3, 0.5, 0.75, 1.0]
    for i, hf in enumerate(hs):
        rh = hf * h
        rx = cx + i * spacing
        ry = y + (h - rh) / 2  # centre vertically
        ax.add_patch(
            plt.Rectangle(
                (rx, ry),
                gap,
                rh,
                facecolor=COL_PROC,
                edgecolor=COL_TEXT,
                linewidth=0.5,
                alpha=0.85,
            )
        )


def draw_icon_sphere_spectrum(ax, x, y, w, h):
    """SFNO -- circle with horizontal arcs suggesting spherical harmonics."""
    cx = x + w / 2
    cy = y + h / 2
    r = min(w * 0.62, h / 2 + 0.006)
    # Outline circle
    circ = plt.Circle((cx, cy), r, facecolor="white", edgecolor=COL_TEXT, linewidth=1.0)
    ax.add_patch(circ)
    # Spherical-harmonic arcs (sinusoid curves clipped to the circle)
    xs = np.linspace(-1, 1, 50)
    for k, amp in [(1, 0.55), (2, 0.30), (3, 0.18)]:
        # vertical position of the curve baseline within the circle
        ys = amp * np.sin(k * np.pi * xs)
        # keep only where inside circle
        mask = (xs**2 + ys**2) < 0.85
        ax.plot(
            cx + xs[mask] * r,
            cy + ys[mask] * r,
            color=COL_PROC,
            lw=0.9,
            alpha=0.85,
        )


def draw_icon_mesh(ax, x, y, w, h):
    """GraphCast -- circle with a few connected nodes (icosahedron-ish patch)."""
    cx = x + w / 2
    cy = y + h / 2
    r = min(w * 0.62, h / 2 + 0.006)
    circ = plt.Circle((cx, cy), r, facecolor="white", edgecolor=COL_TEXT, linewidth=1.0)
    ax.add_patch(circ)
    # Vertices of a small triangulated patch
    pts = np.array(
        [
            [0.0, 0.7],
            [-0.6, 0.2],
            [0.6, 0.2],
            [-0.35, -0.35],
            [0.35, -0.35],
            [0.0, -0.7],
            [0.0, 0.0],
        ]
    )
    px = cx + pts[:, 0] * r
    py = cy + pts[:, 1] * r
    # Edges between selected vertex pairs
    edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (0, 6),
        (5, 6),
    ]
    for i, j in edges:
        ax.plot(
            [px[i], px[j]],
            [py[i], py[j]],
            color=COL_PROC,
            lw=0.6,
            alpha=0.7,
        )
    # Dots at vertices
    ax.scatter(px, py, s=8, color=COL_PROC, edgecolor=COL_TEXT, linewidth=0.4, zorder=3)


def draw_icon_latitude_bands(ax, x, y, w, h):
    """AIFS -- O96 mesh nodes arranged in latitude bands, with horizontal
    'sliding-window' attention strips between adjacent nodes per band. Captures
    the GNN encoder/decoder + along-latitude transformer processor design."""
    n_rows = 3
    n_per_row = 6
    pad_x = 0.10 * w
    pad_y = 0.12 * h
    inner_w = w - 2 * pad_x
    inner_h = h - 2 * pad_y
    row_step = inner_h / (n_rows - 1) if n_rows > 1 else 0
    col_step = inner_w / (n_per_row - 1) if n_per_row > 1 else 0
    for r in range(n_rows):
        cy = y + h - pad_y - r * row_step
        # Faint horizontal band (the sliding attention window)
        ax.plot(
            [x + pad_x, x + pad_x + inner_w],
            [cy, cy],
            color=COL_PROC,
            lw=0.8,
            alpha=0.45,
        )
        # Node dots along the band
        xs = x + pad_x + np.arange(n_per_row) * col_step
        ys = np.full_like(xs, cy, dtype=float)
        ax.scatter(xs, ys, s=5, color=COL_PROC, edgecolor=COL_TEXT, linewidth=0.3, zorder=3)


ICON_DISPATCH = {
    "Aurora": draw_icon_unet,
    "SFNO": draw_icon_sphere_spectrum,
    "GraphCast": draw_icon_mesh,
    "AIFS": draw_icon_latitude_bands,
}


def draw_model_row(ax, y_center, model, n_max_global):
    row_h = ROW_HEIGHT
    y_box_bottom = y_center - row_h / 2

    # ---- Left-column model label block ------------------------------------
    lx = LEFT_COL_X

    # Scale the label-block vertical offsets so multi-row layouts (4+ models)
    # compress vertically and stop overlapping the next row.
    s = _LABEL_SCALE
    LEFT_COL_OFFSET = 0.015 * s

    # Icon above model name (small schematic)
    icon_w = 0.07
    icon_h = 0.05 * s
    icon_x = lx
    icon_y = y_center + 0.030 * s + LEFT_COL_OFFSET
    icon_fn = ICON_DISPATCH.get(model["name"])
    if icon_fn:
        icon_fn(ax, icon_x, icon_y, icon_w, icon_h)

    # Model name (large)
    ax.text(
        lx,
        y_center - 0.005 + LEFT_COL_OFFSET,
        model["name"],
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COL_TEXT,
    )
    # Subtitle (italic, muted)
    ax.text(
        lx,
        y_center - 0.052 * s + LEFT_COL_OFFSET,
        model["subtitle"],
        ha="left",
        va="top",
        fontsize=10,
        color=COL_MUTED,
        style="italic",
    )
    # Total parameter count
    ax.text(
        lx,
        y_center - 0.100 * s + LEFT_COL_OFFSET,
        rf"$N_\mathrm{{total}} = {model['n_total']}$",
        ha="left",
        va="top",
        fontsize=10,
        color=COL_TEXT,
    )

    # ---- Box geometry (within box area) -----------------------------------
    n_total = model["n_total"]
    boxes_meta = []
    x_cursor = 0.0
    gap = 0.018
    for g in model["groups"]:
        w = width_for_n(g["n"], n_max_global)
        sigma = sigma_scaled(n_total, g["n"])
        boxes_meta.append((x_cursor, w, g, sigma))
        x_cursor += w + gap
    used_w = x_cursor - gap  # total width including gaps but no trailing gap

    # Centre boxes within the box area
    shift = BOX_AREA_X + (BOX_AREA_W - used_w) / 2
    boxes_meta = [(rx + shift, w, g, s) for (rx, w, g, s) in boxes_meta]

    # ---- Draw arrows then boxes ------------------------------------------
    for i, (x, w, g, s) in enumerate(boxes_meta):
        if i > 0:
            x_prev, w_prev, _, _ = boxes_meta[i - 1]
            draw_arrow(ax, x_prev + w_prev, x, y_center)

        draw_box(
            ax,
            x,
            y_box_bottom,
            w,
            row_h,
            kind_color(g["kind"]),
            g["key"],
            g["n"],
            s,
            g["detail"],
            highlight=g.get("highlight"),
        )


def main():
    n_max = max(g["n"] for m in MODELS for g in m["groups"])

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Subtle grid background (very light)
    for gx in np.linspace(0.05, 0.95, 19):
        ax.axvline(gx, color=COL_GRID, lw=0.4, alpha=0.3, zorder=0)
    for gy in np.linspace(0.05, 0.95, 19):
        ax.axhline(gy, color=COL_GRID, lw=0.4, alpha=0.3, zorder=0)

    # Title (compact, near top edge)
    _num = {3: "three", 4: "four", 5: "five", 6: "six"}.get(_n_models, str(_n_models))
    ax.text(
        LEFT_COL_X,
        0.995,
        f"Weight perturbation across {_num} AI weather models",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COL_TEXT,
    )

    # Three model rows
    for y, model in zip(ROW_CENTERS, MODELS):
        draw_model_row(ax, y, model, n_max)

    # Colour + glyph legend pinned to the bottom margin
    foot_y = 0.012
    legend_items = [
        ("encoder", COL_ENC),
        ("processor", COL_PROC),
        ("decoder", COL_DEC),
        ("aux / residual", COL_AUX),
    ]
    # Right-align the legend to the bottom model row's rightmost box edge (the
    # row sitting directly above the footer; boxes are centred per row).
    _gap = 0.018
    _bot_groups = MODELS[-1]["groups"]
    _bot_used = sum(width_for_n(g["n"], n_max) for g in _bot_groups) + _gap * (len(_bot_groups) - 1)
    rx = BOX_AREA_X + (BOX_AREA_W + _bot_used) / 2 - 0.015
    for label, colour in reversed(legend_items):
        ax.text(
            rx,
            foot_y,
            label,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=colour,
        )
        rx -= 0.13

    out_dir = Path(__file__).parent
    fig.savefig(out_dir / "perturbation_schematic.svg", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_dir / "perturbation_schematic.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(
        out_dir / "perturbation_schematic.png",
        bbox_inches="tight",
        pad_inches=0.15,
        dpi=220,
    )
    print(f"wrote {out_dir / 'perturbation_schematic.svg'}")
    print(f"wrote {out_dir / 'perturbation_schematic.pdf'}")
    print(f"wrote {out_dir / 'perturbation_schematic.png'}")


if __name__ == "__main__":
    main()
