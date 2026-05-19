#!/usr/bin/env python3
"""Schematic for Phase 3 physics-inspired weight perturbation.

Each of the three deterministic models has a different *mechanism* for
targeting the same physical scale band -- wavelengths >= ~3000-4000 km
(planetary / large-synoptic):

  SFNO       low spherical-harmonic modes (l <= 10) in spectral conv weights
  Aurora     bottleneck Swin layers (encoder_layers.2 + decoder_layers.0)
  GraphCast  long edges of the multi-mesh (refinement levels 0-1)

Output: figures/phase3_schematic.{svg,pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
    }
)

# Colour palette: physics theme uses warmer + atmospheric blues
COL_TARGET = "#D14F2A"  # warm red-orange for the *targeted* coarse-scale region
COL_MUTED_REG = "#A0B5C0"  # cool grey-blue for non-targeted
COL_ACCENT = "#2C5F7C"  # deep atmospheric blue
COL_GLOW = "#F4A53D"  # gold glow on targeted parts
COL_TEXT = "#1F2A36"
COL_FAINT = "#5C6B7A"
COL_GRID = "#E9ECEF"

# Layout
LEFT_COL_X = 0.02
LEFT_COL_W = 0.22
VIZ_AREA_X = LEFT_COL_X + LEFT_COL_W + 0.02
VIZ_AREA_W = 1.0 - VIZ_AREA_X - 0.02
ROW_HEIGHT = 0.21
ROW_CENTERS = [0.74, 0.46, 0.18]


# -- Model-specific visualizations -------------------------------------------


def draw_sfno_spectral_lattice(ax, x, y, w, h):
    """Spectral modes as a centred bar chart.

    Each column = one total wavenumber l. Bar height = number of modes
    at that l (2l+1), centred vertically. Bars with l <= l_cut are the
    targeted (planetary) slice; bars with l > l_cut are faded grey.

    This conveys both the triangular SH structure (taller bars at larger
    l) and the partition into perturbed vs untouched regions.
    """
    lmax_show = 25
    l_cut = 10

    pad_left = 0.050
    pad_right = 0.025
    pad_top = 0.030
    pad_bot = 0.035

    grid_w = w - pad_left - pad_right
    grid_h = h - pad_top - pad_bot

    col_w = grid_w / (lmax_show + 1)
    max_modes = 2 * lmax_show + 1
    # Vertical unit so the tallest bar fills grid_h
    v_unit = grid_h / max_modes

    base_x = x + pad_left
    cy_mid = y + pad_bot + grid_h / 2
    bar_w = col_w * 0.78

    for l_idx in range(lmax_show + 1):
        modes = 2 * l_idx + 1
        bar_h = modes * v_unit
        cx = base_x + l_idx * col_w + col_w / 2
        target = l_idx <= l_cut
        col = COL_TARGET if target else COL_MUTED_REG
        alpha = 0.95 if target else 0.45
        ax.add_patch(
            FancyBboxPatch(
                (cx - bar_w / 2, cy_mid - bar_h / 2),
                bar_w,
                bar_h,
                boxstyle="round,pad=0,rounding_size=0.0035",
                facecolor=col,
                edgecolor=col,
                linewidth=0.4,
                alpha=alpha,
                zorder=3,
            )
        )

    # Vertical dashed threshold between l_cut and l_cut+1
    x_cut = base_x + (l_cut + 1) * col_w
    ax.plot(
        [x_cut, x_cut],
        [y + pad_bot - 0.003, y + pad_bot + grid_h + 0.003],
        color=COL_GLOW,
        lw=1.6,
        linestyle=(0, (4, 2)),
        alpha=0.95,
        zorder=4,
    )
    # Threshold label
    ax.text(
        x_cut + 0.008,
        y + h - pad_top - 0.005,
        f"l ≤ {l_cut}\nλ ≥ 4000 km",
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=COL_TARGET,
        zorder=5,
    )

    # Axis labels
    ax.text(
        x + w / 2,
        y + 0.008,
        "total wavenumber  l   →",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=COL_FAINT,
        style="italic",
    )
    ax.text(
        x + pad_left * 0.25,
        cy_mid,
        "# modes per l  (= 2l+1)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=COL_FAINT,
        style="italic",
        rotation=90,
    )

    # Panel title (top)
    ax.text(
        x + w / 2,
        y + h - 0.012,
        "spectral weight modes  (per SFNO block, ×8 blocks)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COL_TEXT,
        style="italic",
    )


def draw_aurora_unet_bottleneck(ax, x, y, w, h):
    """U-Net cross-section with the bottleneck layers highlighted."""
    pad = 0.018
    pad_top = 0.030  # room for title strip
    pad_bot = 0.030  # room for labels

    inner_w = w - 2 * pad
    inner_h = h - pad_top - pad_bot
    n_levels = 4
    col_w = inner_w / (2 * n_levels - 1)
    heights = [
        inner_h,
        inner_h * 0.72,
        inner_h * 0.50,
        inner_h * 0.30,
        inner_h * 0.50,
        inner_h * 0.72,
        inner_h,
    ]
    labels = ["enc_0", "enc_1", "enc_2", "btnk", "dec_0", "dec_1", "dec_2"]
    targeted_idx = {2, 4}

    base_x = x + pad
    cy = y + pad_bot + inner_h / 2

    for i, (rh, lab) in enumerate(zip(heights, labels)):
        rx = base_x + i * col_w
        ry = cy - rh / 2
        targ = i in targeted_idx
        col = COL_TARGET if targ else COL_MUTED_REG
        alpha = 0.95 if targ else 0.55
        rect = FancyBboxPatch(
            (rx + col_w * 0.10, ry),
            col_w * 0.80,
            rh,
            boxstyle="round,pad=0,rounding_size=0.008",
            facecolor=col,
            edgecolor=col,
            linewidth=0.4,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(rect)
        if targ:
            glow = FancyBboxPatch(
                (rx + col_w * 0.06, ry - 0.006),
                col_w * 0.88,
                rh + 0.012,
                boxstyle="round,pad=0,rounding_size=0.010",
                facecolor="none",
                edgecolor=COL_GLOW,
                linewidth=1.5,
                alpha=0.9,
                zorder=1,
            )
            ax.add_patch(glow)
        ax.text(
            rx + col_w / 2,
            y + 0.008,
            lab,
            ha="center",
            va="bottom",
            fontsize=7,
            color=COL_TEXT if targ else COL_FAINT,
            fontweight="bold" if targ else "normal",
        )

    # Faint arrows between consecutive layers
    for i in range(len(labels) - 1):
        rx = base_x + i * col_w + col_w * 0.90
        rx2 = base_x + (i + 1) * col_w + col_w * 0.10
        ax.annotate(
            "",
            xy=(rx2, cy),
            xytext=(rx, cy),
            arrowprops=dict(arrowstyle="-|>", color=COL_FAINT, lw=0.6, alpha=0.5),
            zorder=1,
        )

    # Panel title (top, no overlap)
    ax.text(
        x + w / 2,
        y + h - 0.012,
        "Swin-3D U-Net depth  (block widths illustrate spatial resolution)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COL_TEXT,
        style="italic",
    )

    # Highlighted regions get one inline annotation each, NOT crossing the title.
    enc2_cx = base_x + 2 * col_w + col_w / 2
    dec0_cx = base_x + 4 * col_w + col_w / 2
    ax.text(
        enc2_cx,
        cy + heights[2] / 2 + 0.013,
        "encoder_layers.2",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color=COL_TARGET,
    )
    ax.text(
        dec0_cx,
        cy + heights[4] / 2 + 0.013,
        "decoder_layers.0",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color=COL_TARGET,
    )


def draw_graphcast_multimesh(ax, x, y, w, h):
    """Icosahedral multi-mesh: long edges (lvl 0-1) highlighted, multi-level rings."""
    pad_top = 0.030  # room for panel title
    pad_bot = 0.018

    cx_g = x + w / 2
    cy_g = y + pad_bot + (h - pad_top - pad_bot) / 2

    r_outer = min(w - 0.04, h - pad_top - pad_bot) / 2 - 0.005
    sphere = Circle(
        (cx_g, cy_g),
        r_outer,
        facecolor="#F4F8FB",
        edgecolor=COL_ACCENT,
        linewidth=0.8,
        alpha=0.9,
    )
    ax.add_patch(sphere)

    # --- Level 0: 5 vertices in pentagon, edges between them = ~6700 km ---
    n0 = 5
    v0 = np.column_stack(
        [
            cx_g + r_outer * 0.85 * np.cos(2 * np.pi * np.arange(n0) / n0 + np.pi / 2),
            cy_g + r_outer * 0.85 * np.sin(2 * np.pi * np.arange(n0) / n0 + np.pi / 2),
        ]
    )

    # All-to-all edges among level-0 vertices (longest edges in the mesh)
    for i in range(n0):
        for j in range(i + 1, n0):
            ax.plot(
                [v0[i, 0], v0[j, 0]],
                [v0[i, 1], v0[j, 1]],
                color=COL_TARGET,
                lw=2.6,
                alpha=0.95,
                zorder=4,
                solid_capstyle="round",
            )

    # --- Level 1: midpoints of level-0 edges = ~3300 km ---
    mid_pairs = [(i, (i + 1) % n0) for i in range(n0)]
    v1 = np.array([(v0[i] + v0[j]) / 2 for i, j in mid_pairs])
    # Edges between adjacent level-1 vertices
    for i in range(n0):
        a, b = v1[i], v1[(i + 1) % n0]
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=COL_GLOW,
            lw=1.8,
            alpha=0.95,
            zorder=3,
        )
    # Level-1 edges back to nearest level-0 vertex (also long)
    for i in range(n0):
        ax.plot(
            [v1[i, 0], v0[i, 0]],
            [v1[i, 1], v0[i, 1]],
            color=COL_GLOW,
            lw=1.5,
            alpha=0.85,
            zorder=3,
        )

    # --- Level 2+ : background scatter of short-edge mesh (untouched) ---
    rng = np.random.default_rng(7)
    n_bg = 70
    angles = rng.uniform(0, 2 * np.pi, n_bg)
    radii = rng.uniform(0.15, 0.92, n_bg) * r_outer
    bg_x = cx_g + radii * np.cos(angles)
    bg_y = cy_g + radii * np.sin(angles)
    # Connect each background point to its 2 nearest neighbours
    pts = np.column_stack([bg_x, bg_y])
    for i in range(n_bg):
        dists = np.linalg.norm(pts - pts[i], axis=1)
        dists[i] = np.inf
        for j in np.argsort(dists)[:2]:
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                color=COL_MUTED_REG,
                lw=0.35,
                alpha=0.45,
                zorder=2,
            )
    ax.scatter(bg_x, bg_y, s=4, color=COL_MUTED_REG, alpha=0.6, edgecolors="none", zorder=2)

    # Highlighted nodes on top
    ax.scatter(
        v1[:, 0], v1[:, 1], s=40, color=COL_GLOW, edgecolors="white", linewidths=0.6, zorder=5
    )
    ax.scatter(
        v0[:, 0], v0[:, 1], s=70, color=COL_TARGET, edgecolors="white", linewidths=0.6, zorder=6
    )

    # Panel title (top of panel)
    ax.text(
        x + w / 2,
        y + h - 0.012,
        "Multi-mesh icosahedron  (longer edges = coarser scales)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COL_TEXT,
        style="italic",
    )

    # Single compact edge-length legend in the bottom-right corner of panel
    ax.text(
        x + w - 0.015,
        y + pad_bot + 0.005,
        "edge length:  level 0 ~ 6700 km    level 1 ~ 3300 km    level 6 ~ 100 km",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=COL_FAINT,
        style="italic",
    )


# -- Architectural icon (reused from Phase 2 figure for consistency) ---------


def draw_icon_unet(ax, x, y, w, h):
    n = 7
    pad = 0.005
    gap = (w - 2 * pad) / (n + (n - 1) * 0.25)
    spacing = gap * 1.25
    cx = x + pad
    hs = [1.0, 0.75, 0.5, 0.3, 0.5, 0.75, 1.0]
    for i, hf in enumerate(hs):
        rh = hf * h
        rx = cx + i * spacing
        ry = y + (h - rh) / 2
        ax.add_patch(
            plt.Rectangle(
                (rx, ry),
                gap,
                rh,
                facecolor=COL_ACCENT,
                edgecolor=COL_TEXT,
                linewidth=0.5,
                alpha=0.85,
            )
        )


def draw_icon_sphere(ax, x, y, w, h):
    cx = x + w / 2
    cy = y + h / 2
    r = min(w, h) / 2 - 0.003
    circ = plt.Circle((cx, cy), r, facecolor="white", edgecolor=COL_TEXT, linewidth=1.0)
    ax.add_patch(circ)
    xs = np.linspace(-1, 1, 50)
    for k, amp in [(1, 0.55), (2, 0.30), (3, 0.18)]:
        ys = amp * np.sin(k * np.pi * xs)
        mask = (xs**2 + ys**2) < 0.85
        ax.plot(cx + xs[mask] * r, cy + ys[mask] * r, color=COL_ACCENT, lw=0.9, alpha=0.85)


def draw_icon_mesh(ax, x, y, w, h):
    cx = x + w / 2
    cy = y + h / 2
    r = min(w, h) / 2 - 0.003
    circ = plt.Circle((cx, cy), r, facecolor="white", edgecolor=COL_TEXT, linewidth=1.0)
    ax.add_patch(circ)
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
        ax.plot([px[i], px[j]], [py[i], py[j]], color=COL_ACCENT, lw=0.6, alpha=0.7)
    ax.scatter(px, py, s=6, color=COL_ACCENT, edgecolor=COL_TEXT, linewidth=0.4, zorder=3)


ICONS = {"Aurora": draw_icon_unet, "SFNO": draw_icon_sphere, "GraphCast": draw_icon_mesh}


# -- Per-row layout -----------------------------------------------------------


MODELS = [
    {
        "name": "SFNO",
        "subtitle": "Spherical Fourier Neural Operator",
        "mechanism": "spectral weight sub-slice perturbation",
        "target": "l ≤ 10 modes of every block's filter weight",
        "scale": "λ ≥ 4000 km  (planetary)",
        "viz": draw_sfno_spectral_lattice,
    },
    {
        "name": "Aurora",
        "subtitle": "Swin-Transformer 3D U-Net",
        "mechanism": "bottleneck-layer weight perturbation",
        "target": "encoder_layers.2 + decoder_layers.0",
        "scale": "λ ≈ 1000–3000 km  (large synoptic)",
        "viz": draw_aurora_unet_bottleneck,
    },
    {
        "name": "GraphCast",
        "subtitle": "Multi-Mesh Graph Neural Network",
        "mechanism": "runtime hook on edge embeddings",
        "target": "edges at mesh refinement levels 0–1",
        "scale": "λ ≥ 3300 km  (planetary)",
        "viz": draw_graphcast_multimesh,
    },
]


def draw_row(ax, y_center, model):
    row_h = ROW_HEIGHT
    y_box_bottom = y_center - row_h / 2

    # Left column
    lx = LEFT_COL_X
    icon_fn = ICONS.get(model["name"])
    if icon_fn:
        icon_fn(ax, lx, y_center + 0.038, 0.07, 0.05)
    ax.text(
        lx,
        y_center,
        model["name"],
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=COL_TEXT,
    )
    ax.text(
        lx,
        y_center - 0.034,
        model["subtitle"],
        ha="left",
        va="top",
        fontsize=8,
        color=COL_FAINT,
        style="italic",
    )
    ax.text(
        lx,
        y_center - 0.062,
        "Mechanism:",
        ha="left",
        va="top",
        fontsize=8,
        color=COL_TEXT,
        fontweight="bold",
    )
    ax.text(
        lx + 0.0,
        y_center - 0.080,
        model["mechanism"],
        ha="left",
        va="top",
        fontsize=8,
        color=COL_ACCENT,
    )
    ax.text(
        lx,
        y_center - 0.103,
        "Target slice:",
        ha="left",
        va="top",
        fontsize=8,
        color=COL_TEXT,
        fontweight="bold",
    )
    ax.text(
        lx,
        y_center - 0.121,
        model["target"],
        ha="left",
        va="top",
        fontsize=8,
        color=COL_TARGET,
    )

    # Big visualisation panel on the right
    viz_pad = 0.012
    bg = FancyBboxPatch(
        (VIZ_AREA_X, y_box_bottom),
        VIZ_AREA_W,
        row_h,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="white",
        edgecolor=COL_GRID,
        linewidth=0.8,
        zorder=1,
    )
    ax.add_patch(bg)
    model["viz"](
        ax,
        VIZ_AREA_X + viz_pad,
        y_box_bottom + viz_pad,
        VIZ_AREA_W - 2 * viz_pad,
        row_h - 2 * viz_pad,
    )

    # Scale annotation on top-right corner of panel
    ax.text(
        VIZ_AREA_X + VIZ_AREA_W - 0.012,
        y_box_bottom + row_h - 0.0,
        model["scale"],
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COL_TARGET,
    )


def main():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Very faint grid
    for gx in np.linspace(0.05, 0.95, 19):
        ax.axvline(gx, color=COL_GRID, lw=0.4, alpha=0.25, zorder=0)
    for gy in np.linspace(0.05, 0.95, 19):
        ax.axhline(gy, color=COL_GRID, lw=0.4, alpha=0.25, zorder=0)

    # Title
    ax.text(
        LEFT_COL_X,
        0.985,
        "Physics-inspired coarse-scale perturbation",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=COL_TEXT,
    )
    ax.text(
        LEFT_COL_X,
        0.940,
        "Phase 3 ablation: target the parameters responsible for "
        r"wavelengths $\lambda \gtrsim 3000$ km (planetary / large synoptic).  "
        "Each model uses a different mechanism for the same physical objective.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=COL_FAINT,
    )

    for y, model in zip(ROW_CENTERS, MODELS):
        draw_row(ax, y, model)

    # Bottom legend
    foot_y = 0.012
    swatches = [
        ("targeted (perturbed)", COL_TARGET),
        ("intermediate scale", COL_GLOW),
        ("untouched (small-scale)", COL_MUTED_REG),
    ]
    lx = LEFT_COL_X
    for label, col in swatches:
        # Color dot
        ax.scatter(lx, foot_y + 0.012, s=80, color=col, edgecolor="white", linewidth=0.5, zorder=5)
        ax.text(
            lx + 0.015,
            foot_y,
            label,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=COL_TEXT,
        )
        lx += 0.20

    out_dir = Path(__file__).parent
    fig.savefig(out_dir / "phase3_schematic.svg", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_dir / "phase3_schematic.pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out_dir / "phase3_schematic.png", bbox_inches="tight", pad_inches=0.15, dpi=220)
    print(f"wrote {out_dir / 'phase3_schematic.svg'}")
    print(f"wrote {out_dir / 'phase3_schematic.pdf'}")
    print(f"wrote {out_dir / 'phase3_schematic.png'}")


if __name__ == "__main__":
    main()
