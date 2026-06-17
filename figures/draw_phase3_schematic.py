#!/usr/bin/env python3
"""Schematic for Phase 3 physics-inspired weight perturbation.

Each of the three deterministic models has a different *mechanism* for
targeting the same physical scale band -- wavelengths >= ~3000-4000 km
(planetary / large-synoptic):

  SFNO       low spherical-harmonic modes (l <= 10) in spectral conv weights
  Aurora     coarse encoder block only (encoder_layers.2)
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
LEFT_COL_W = 0.21
VIZ_AREA_X = LEFT_COL_X + LEFT_COL_W + 0.020
VIZ_AREA_W = 1.0 - VIZ_AREA_X - 0.015
ROW_HEIGHT = 0.225
ROW_CENTERS = [0.755, 0.475, 0.195]


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

    # Panel caption: what the bars mean (no architecture name repetition)
    ax.text(
        x + w / 2,
        y + h - 0.012,
        "modes per block × 8 blocks  ·  bar height = 2l+1 modes per l",
        ha="center",
        va="top",
        fontsize=8.5,
        color=COL_TEXT,
        style="italic",
    )


def draw_aurora_unet_bottleneck(ax, x, y, w, h):
    """U-Net cross-section with the deepest *encoder* block highlighted.

    Aurora has 3 encoder + 3 decoder Swin-3D layers. Channels double per
    downsample (512 / 1024 / 2048), spatial resolution halves per stage
    (1° / 2° / 4° tokens given the 4x4 patch embed on a 0.25° input grid).
    Only `encoder_layers.2` (enc_2) is the Phase 3 target. Decoder side
    not perturbed: Phase 2 showed encoder perturbation produces much
    larger ensemble spread than decoder, so doubling down on the encoder
    side is cleaner than mixing dec_0 in.
    """
    pad = 0.018
    pad_top = 0.024
    pad_bot = 0.052  # taller bottom to fit two-line block annotations

    inner_w = w - 2 * pad
    inner_h = h - pad_top - pad_bot
    # 6 blocks -- only enc_2 is the Phase 3 target
    blocks = [
        {"name": "enc_0", "ch": 512, "res_deg": 1, "res_km": 112, "frac": 1.00},
        {"name": "enc_1", "ch": 1024, "res_deg": 2, "res_km": 224, "frac": 0.72},
        {"name": "enc_2", "ch": 2048, "res_deg": 4, "res_km": 448, "frac": 0.42, "tgt": True},
        {"name": "dec_0", "ch": 2048, "res_deg": 4, "res_km": 448, "frac": 0.42},
        {"name": "dec_1", "ch": 1024, "res_deg": 2, "res_km": 224, "frac": 0.72},
        {"name": "dec_2", "ch": 512, "res_deg": 1, "res_km": 112, "frac": 1.00},
    ]
    n = len(blocks)
    col_w = inner_w / n
    base_x = x + pad
    cy = y + pad_bot + inner_h / 2

    for i, blk in enumerate(blocks):
        rx = base_x + i * col_w
        rh = blk["frac"] * inner_h
        ry = cy - rh / 2
        targ = blk.get("tgt", False)
        col = COL_TARGET if targ else COL_MUTED_REG
        alpha = 0.95 if targ else 0.55
        ax.add_patch(
            FancyBboxPatch(
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
        )
        if targ:
            ax.add_patch(
                FancyBboxPatch(
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
            )
        # Single-line annotation under the block: name + compact dims
        ax.text(
            rx + col_w / 2,
            y + pad_bot - 0.005,
            blk["name"],
            ha="center",
            va="top",
            fontsize=7.5,
            color=COL_TARGET if targ else COL_TEXT,
            fontweight="bold" if targ else "normal",
        )
        ax.text(
            rx + col_w / 2,
            y + pad_bot - 0.020,
            f"{blk['ch']}c · {blk['res_deg']}°",
            ha="center",
            va="top",
            fontsize=6.8,
            color=COL_FAINT,
        )

    # Faint arrows between consecutive blocks
    for i in range(n - 1):
        rx = base_x + i * col_w + col_w * 0.90
        rx2 = base_x + (i + 1) * col_w + col_w * 0.10
        ax.annotate(
            "",
            xy=(rx2, cy),
            xytext=(rx, cy),
            arrowprops=dict(arrowstyle="-|>", color=COL_FAINT, lw=0.6, alpha=0.5),
            zorder=1,
        )

    # Panel caption (single line; bottleneck info compressed into it)
    ax.text(
        x + w / 2,
        y + h - 0.012,
        "block widths ∝ spatial resolution  ·  c = channels  ·  " "enc_2 attn window ≈ 5000 km",
        ha="center",
        va="top",
        fontsize=8.0,
        color=COL_TEXT,
        style="italic",
    )


def _icosahedron_verts_3d() -> np.ndarray:
    """Return the 12 vertices of a unit-radius regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=float,
    )
    return verts / np.linalg.norm(verts[0])


def _icosahedron_edges() -> list[tuple[int, int]]:
    """Return the 30 edges of an icosahedron (pairs of indices into the 12 verts)."""
    verts = _icosahedron_verts_3d()
    # Each edge length = nearest-neighbour distance
    edges = []
    for i in range(12):
        dists = np.linalg.norm(verts - verts[i], axis=1)
        # exactly 5 nearest neighbours per vertex (icosahedron property)
        nbrs = np.argsort(dists)[1:6]
        for j in nbrs:
            if (i, int(j)) not in edges and (int(j), i) not in edges:
                edges.append((i, int(j)))
    return edges


def _icosahedron_subdivided() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return (42 verts, 120 edges) for one Loop-style subdivision of the icosahedron.

    Each original edge gets a new midpoint vertex (normalised to the unit
    sphere). Each original triangle becomes 4 smaller triangles whose new
    edges form the next refinement level.
    """
    verts0 = _icosahedron_verts_3d()
    edges0 = _icosahedron_edges()

    # 20 triangular faces of an icosahedron
    nbrs = {i: set() for i in range(12)}
    for a, b in edges0:
        nbrs[a].add(b)
        nbrs[b].add(a)
    faces = set()
    for a, b in edges0:
        for c in nbrs[a] & nbrs[b]:
            faces.add(tuple(sorted([a, b, c])))
    faces = list(faces)

    # Midpoint of each original edge, normalised back to the unit sphere
    mid_index = {}
    mid_verts = []
    for a, b in edges0:
        key = (min(a, b), max(a, b))
        m = (verts0[a] + verts0[b]) / 2
        m = m / np.linalg.norm(m)
        mid_index[key] = 12 + len(mid_verts)
        mid_verts.append(m)
    verts1 = np.vstack([verts0, np.array(mid_verts)])

    # Each original triangle (a, b, c) -> 4 sub-triangles with new edges.
    new_edges = set()

    def _add(p, q):
        new_edges.add((min(p, q), max(p, q)))

    for a, b, c in faces:
        mab = mid_index[(min(a, b), max(a, b))]
        mbc = mid_index[(min(b, c), max(b, c))]
        mca = mid_index[(min(c, a), max(c, a))]
        # 4 sub-triangles: (a, mab, mca), (mab, b, mbc), (mca, mbc, c),
        # (mab, mbc, mca). Each contributes 3 edges (some shared).
        for tri in [(a, mab, mca), (mab, b, mbc), (mca, mbc, c), (mab, mbc, mca)]:
            for i in range(3):
                _add(tri[i], tri[(i + 1) % 3])
    return verts1, list(new_edges)


def _fibonacci_sphere_points(n: int) -> np.ndarray:
    """Return n quasi-uniform (x, y, z) points on a unit sphere (Fibonacci spiral)."""
    if n <= 0:
        return np.zeros((0, 3))
    indices = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * indices / n)  # latitude
    theta = np.pi * (1 + np.sqrt(5)) * indices  # longitude
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


def _orthographic_project(verts: np.ndarray, view_z: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the sphere slightly for a 3/4 view, then project to (x, y).

    Returns (2D coords, z-depth in [-1, 1]).
    """
    # Rotate so the camera looks down a slightly off-axis direction. This
    # avoids the degenerate "pentagon dead-on" view that hides the icosahedral
    # face pattern.
    theta = 0.32  # ~18° tilt around x-axis
    R = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    rotated = verts @ R.T
    return rotated[:, :2], rotated[:, 2]


def _draw_single_mesh_globe(ax, cx, cy, r, level_k, edge_col, edge_lw, node_col, node_size, alpha):
    """Draw one small mesh globe at the given centre / radius.

    For levels 0 and 1: use the exact icosahedron / first-subdivision
    geometry so the pentagonal/hexagonal pattern is visible.
    For higher levels: Fibonacci-uniform points + Delaunay triangulation
    (cheap visual proxy; individual edges aren't readable anyway).
    """
    from matplotlib.tri import Triangulation

    # Sphere outline
    ax.add_patch(
        Circle(
            (cx, cy),
            r,
            facecolor="#F4F8FB",
            edgecolor=COL_ACCENT,
            linewidth=0.7,
            alpha=0.95,
            zorder=2,
        )
    )

    if level_k == 0:
        verts3d = _icosahedron_verts_3d()
        edges = _icosahedron_edges()
        verts2d, zs = _orthographic_project(verts3d)
        px = cx + verts2d[:, 0] * r * 0.94
        py = cy + verts2d[:, 1] * r * 0.94
        # Edges: draw back-hemisphere edges faded
        for a, b in edges:
            z_avg = (zs[a] + zs[b]) / 2
            front = z_avg > -0.05
            a_alpha = alpha if front else alpha * 0.35
            ax.plot(
                [px[a], px[b]],
                [py[a], py[b]],
                color=edge_col,
                lw=edge_lw if front else edge_lw * 0.7,
                alpha=a_alpha,
                zorder=3 if front else 2.5,
                solid_capstyle="round",
            )
        if node_size > 0:
            for i in range(len(verts3d)):
                front = zs[i] > -0.05
                ax.scatter(
                    px[i],
                    py[i],
                    s=node_size if front else node_size * 0.55,
                    color=node_col,
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=alpha if front else alpha * 0.5,
                    zorder=5 if front else 4,
                )

    elif level_k == 1:
        verts3d, edges = _icosahedron_subdivided()
        verts2d, zs = _orthographic_project(verts3d)
        px = cx + verts2d[:, 0] * r * 0.94
        py = cy + verts2d[:, 1] * r * 0.94
        for a, b in edges:
            z_avg = (zs[a] + zs[b]) / 2
            front = z_avg > -0.08
            a_alpha = alpha if front else alpha * 0.30
            ax.plot(
                [px[a], px[b]],
                [py[a], py[b]],
                color=edge_col,
                lw=edge_lw if front else edge_lw * 0.7,
                alpha=a_alpha,
                zorder=3 if front else 2.5,
                solid_capstyle="round",
            )
        if node_size > 0:
            for i in range(len(verts3d)):
                front = zs[i] > -0.08
                ax.scatter(
                    px[i],
                    py[i],
                    s=node_size if front else node_size * 0.55,
                    color=node_col,
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=alpha if front else alpha * 0.5,
                    zorder=5 if front else 4,
                )

    else:
        # Higher refinement: visual approximation only
        n_draw = {2: 80, 3: 140, 4: 240, 5: 420}.get(level_k, 100)
        sphere_pts = _fibonacci_sphere_points(n_draw)
        verts2d, zs = _orthographic_project(sphere_pts)
        # Keep only front hemisphere for cleaner look
        front_mask = zs > -0.05
        px = cx + verts2d[front_mask, 0] * r * 0.94
        py = cy + verts2d[front_mask, 1] * r * 0.94
        if len(px) >= 3:
            tri = Triangulation(px, py)
            max_edge = 4.0 * r / np.sqrt(max(len(px), 1))
            for t in tri.triangles:
                for i in range(3):
                    a, b = t[i], t[(i + 1) % 3]
                    d = np.hypot(px[a] - px[b], py[a] - py[b])
                    if d > max_edge or d > r * 0.85:
                        continue
                    ax.plot(
                        [px[a], px[b]],
                        [py[a], py[b]],
                        color=edge_col,
                        lw=edge_lw,
                        alpha=alpha,
                        zorder=3,
                        solid_capstyle="round",
                    )
        if node_size > 0:
            ax.scatter(
                px,
                py,
                s=node_size,
                color=node_col,
                edgecolors="white",
                linewidths=0.35,
                alpha=alpha,
                zorder=4,
            )


def draw_graphcast_multimesh(ax, x, y, w, h):
    """Row of small icosahedral mesh-refinement globes.

    Visual quote of GraphCast paper figure 2: a series of small spheres
    showing successive refinement of the icosahedron, from coarse (level 0,
    targeted) to fine (level 5, untouched).
    """
    pad_x = 0.012
    pad_top = 0.012
    pad_bot = 0.040  # leave room for per-globe labels

    inner_x = x + pad_x
    inner_w = w - 2 * pad_x
    inner_h = h - pad_top - pad_bot

    # 6 refinement levels (0..5). Level k has roughly 10*4^k + 2 vertices,
    # but we cap n_points for visual clarity at higher levels.
    levels = [
        {"k": 0, "n_actual": 12, "n_draw": 12, "edge_km": "~6700"},
        {"k": 1, "n_actual": 42, "n_draw": 38, "edge_km": "~3300"},
        {"k": 2, "n_actual": 162, "n_draw": 80, "edge_km": "~1700"},
        {"k": 3, "n_actual": 642, "n_draw": 140, "edge_km": "~830"},
        {"k": 4, "n_actual": 2562, "n_draw": 240, "edge_km": "~410"},
        {"k": 5, "n_actual": 10242, "n_draw": 420, "edge_km": "~210"},
    ]
    n_globes = len(levels)
    gap_frac = 0.06  # gap between globes as fraction of globe diameter
    globe_d = inner_w / (n_globes * (1 + gap_frac) - gap_frac)
    globe_d = min(globe_d, inner_h * 0.95)
    r = globe_d / 2
    gap = globe_d * gap_frac

    # Centre the row vertically and horizontally
    total_w = n_globes * globe_d + (n_globes - 1) * gap
    x_start = inner_x + (inner_w - total_w) / 2
    cy = y + pad_bot + inner_h / 2

    targeted_levels = {0, 1}

    for i, lvl in enumerate(levels):
        cx = x_start + r + i * (globe_d + gap)
        target = lvl["k"] in targeted_levels
        if lvl["k"] == 0:
            edge_col, edge_lw, node_col, node_sz = COL_TARGET, 2.4, COL_TARGET, 22
            alpha = 0.95
        elif lvl["k"] == 1:
            edge_col, edge_lw, node_col, node_sz = COL_GLOW, 1.7, COL_GLOW, 12
            alpha = 0.95
        else:
            edge_col, edge_lw, node_col, node_sz = COL_MUTED_REG, 0.45, COL_MUTED_REG, 2
            alpha = 0.55

        _draw_single_mesh_globe(
            ax,
            cx,
            cy,
            r,
            lvl["k"],
            edge_col,
            edge_lw,
            node_col,
            node_sz,
            alpha,
        )

        # Per-globe label below
        ax.text(
            cx,
            cy - r - 0.011,
            f"level {lvl['k']}",
            ha="center",
            va="top",
            fontsize=8.0,
            fontweight="bold" if target else "normal",
            color=COL_TARGET if lvl["k"] == 0 else (COL_GLOW if lvl["k"] == 1 else COL_TEXT),
        )
        ax.text(
            cx,
            cy - r - 0.028,
            f"{lvl['n_actual']} nodes\nλ {lvl['edge_km']} km",
            ha="center",
            va="top",
            fontsize=7,
            color=COL_FAINT,
            linespacing=1.2,
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
        "scale": "λ ≥ 4000 km  (upper synoptic → planetary)",
        "viz": draw_sfno_spectral_lattice,
    },
    {
        "name": "Aurora",
        "subtitle": "Swin-Transformer 3D U-Net",
        "mechanism": "U-net-bottom weight perturbation",
        "target": "encoder_layers.2 only",
        "scale": "λ ≈ 0.5–5 Mm  (synoptic → planetary)",
        "viz": draw_aurora_unet_bottleneck,
    },
    {
        "name": "GraphCast",
        "subtitle": "Multi-Mesh Graph Neural Network",
        "mechanism": "activation hook on coarse mesh nodes",
        "target": "first 42 mesh nodes (levels 0–1)",
        "scale": "λ ≥ 3300 km  (upper synoptic → planetary)",
        "viz": draw_graphcast_multimesh,
    },
]


def draw_row(ax, y_center, model):
    row_h = ROW_HEIGHT
    y_box_bottom = y_center - row_h / 2
    y_box_top = y_center + row_h / 2

    # Left column -- top-aligned with the viz panel.
    lx = LEFT_COL_X
    # Icon sits just above the model name, near the panel top.
    icon_fn = ICONS.get(model["name"])
    if icon_fn:
        icon_fn(ax, lx, y_box_top - 0.058, 0.07, 0.05)

    # Model name aligned just below the icon.
    name_y = y_box_top - 0.07
    ax.text(
        lx,
        name_y,
        model["name"],
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=COL_TEXT,
    )
    ax.text(
        lx,
        name_y - 0.030,
        model["subtitle"],
        ha="left",
        va="top",
        fontsize=8,
        color=COL_FAINT,
        style="italic",
    )
    ax.text(
        lx,
        name_y - 0.060,
        "Mechanism:",
        ha="left",
        va="top",
        fontsize=8,
        color=COL_TEXT,
        fontweight="bold",
    )
    ax.text(
        lx,
        name_y - 0.077,
        model["mechanism"],
        ha="left",
        va="top",
        fontsize=8,
        color=COL_ACCENT,
    )
    ax.text(
        lx,
        name_y - 0.100,
        "Target slice:",
        ha="left",
        va="top",
        fontsize=8,
        color=COL_TEXT,
        fontweight="bold",
    )
    ax.text(
        lx,
        name_y - 0.117,
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
        r"Phase 3 ablation: target the parameters at wavelengths "
        r"$\lambda \gtrsim 3000$ km (upper synoptic and above).  "
        "A different mechanism per model, same physical objective.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=COL_FAINT,
    )

    for y, model in zip(ROW_CENTERS, MODELS):
        draw_row(ax, y, model)

    # Bottom legend -- left-aligned with the viz panels, not the figure edge
    foot_y = 0.012
    swatches = [
        ("targeted (perturbed)", COL_TARGET),
        ("intermediate scale", COL_GLOW),
        ("untouched (small-scale)", COL_MUTED_REG),
    ]
    lx = VIZ_AREA_X
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
