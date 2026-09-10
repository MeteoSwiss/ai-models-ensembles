#!/usr/bin/env python3
"""Generate the .dc.html artboards + canvas.json for the 20-min SPW talk."""

import json
import pathlib

HERE = pathlib.Path(__file__).parent

# --- design tokens -----------------------------------------------------------
GROUND, WASH, CARD = "#F7F5F1", "#EFEAE1", "#FCFBF8"
INK, MUTED, FAINT = "#16181C", "#5F6570", "#8A8478"
RULE, RULE2 = "#DCD6CB", "#E6E0D5"

AURORA, GRAPHCAST, SFNO, AIFS = "#E67E22", "#27AE60", "#2980B9", "#8E44AD"
AIFSENS, ATLAS, FCN3, IFS = "#8B5A2B", "#C0392B", "#D4A017", "#7F8C8D"

DISPLAY = "'Space Grotesk','Helvetica Neue',Helvetica,Arial,sans-serif"
BODY = "'IBM Plex Sans','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "'IBM Plex Mono','SF Mono',Menlo,Consolas,monospace"

FONTS = (
    "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700"
    "&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
)

TALK = "Stochastically Perturbed Weights &middot; Adamov, Fuhrer, Knutti, Schemm"


# --- helpers -----------------------------------------------------------------
def chip(colour, label, size=11):
    return (
        f'<span style="display:inline-flex;align-items:center;gap:8px">'
        f'<span style="width:{size}px;height:{size}px;background:{colour};'
        f'border-radius:2px;display:inline-block;flex:none"></span>'
        f"<span>{label}</span></span>"
    )


def card(inner, accent=None, pad=22, bg=CARD, grow=True):
    border = f"2px solid {accent}" if accent else f"1px solid {RULE2}"
    g = "flex-grow:1;" if grow else ""
    return (
        f'<div style="background:{bg};border:{border};padding:{pad}px;'
        f'display:flex;flex-direction:column;gap:10px;{g}">{inner}</div>'
    )


def card_title(text, colour=INK, size=20):
    return (
        f'<div style="font-family:{DISPLAY};font-weight:600;font-size:{size}px;'
        f'color:{colour};letter-spacing:-0.01em">{text}</div>'
    )


def p(text, size=17, colour=MUTED, lh=1.45, weight=400):
    return (
        f'<div style="font-size:{size}px;line-height:{lh};color:{colour};'
        f'font-weight:{weight};text-wrap:pretty">{text}</div>'
    )


def stat(value, label, colour=INK, vsize=46, lsize=14):
    return (
        f'<div style="display:flex;flex-direction:column;gap:4px">'
        f'<div style="font-family:{DISPLAY};font-weight:700;font-size:{vsize}px;'
        f'line-height:1;color:{colour};letter-spacing:-0.02em">{value}</div>'
        f'<div style="font-family:{MONO};font-size:{lsize}px;color:{FAINT};'
        f'letter-spacing:0.04em">{label}</div></div>'
    )


def bullets(items, size=18, gap=13, colour=MUTED, marker=INK):
    rows = "".join(
        f'<div style="display:flex;gap:13px;align-items:flex-start">'
        f'<span style="width:6px;height:6px;background:{marker};border-radius:50%;'
        f'margin-top:{int(size*0.62)}px;flex:none"></span>'
        f'<span style="font-size:{size}px;line-height:1.45;color:{colour};'
        f'text-wrap:pretty">{i}</span></div>'
        for i in items
    )
    return f'<div style="display:flex;flex-direction:column;gap:{gap}px">{rows}</div>'


def figure(src, caption=None, maxh=None, width="100%"):
    mh = f"max-height:{maxh}px;" if maxh else ""
    img = (
        f'<img src="{src}" style="width:{width};height:auto;{mh}object-fit:contain;'
        f'display:block;margin:0 auto" alt="" />'
    )
    if caption:
        img += (
            f'<div style="font-family:{MONO};font-size:12px;color:{FAINT};'
            f'margin-top:10px;letter-spacing:0.02em">{caption}</div>'
        )
    return (
        f'<div style="display:flex;flex-direction:column;justify-content:center;'
        f'flex-grow:1;min-height:0">{img}</div>'
    )


def table(head, rows, widths, size=16, head_size=12, rowpad=9):
    ths = "".join(
        f'<div style="flex:{w};font-family:{MONO};font-size:{head_size}px;color:{FAINT};'
        f'letter-spacing:0.06em;text-transform:uppercase;'
        f'text-align:{"right" if i else "left"}">{h}</div>'
        for i, (h, w) in enumerate(zip(head, widths))
    )
    out = [
        f'<div style="display:flex;gap:16px;padding-bottom:9px;'
        f'border-bottom:1px solid {RULE}">{ths}</div>'
    ]
    for cells, style in rows:
        tds = "".join(
            f'<div style="flex:{w};font-size:{size}px;color:{INK};'
            f'font-family:{MONO if i else BODY};'
            f'font-weight:{style.get("weight", 400)};'
            f'text-align:{"right" if i else "left"}">{c}</div>'
            for i, (c, w) in enumerate(zip(cells, widths))
        )
        bg = style.get("bg", "transparent")
        out.append(
            f'<div style="display:flex;gap:16px;align-items:center;padding:{rowpad}px 0;'
            f'border-bottom:1px solid {RULE2};background:{bg}">{tds}</div>'
        )
    return f'<div style="display:flex;flex-direction:column">{"".join(out)}</div>'


def group_label(text):
    return (
        f'<div style="font-family:{MONO};font-size:12px;color:{FAINT};'
        f'letter-spacing:0.08em;text-transform:uppercase;padding-top:12px">{text}</div>'
    )


# --- slide shell -------------------------------------------------------------
def slide(n, eyebrow, title, body, sub=None, title_size=41, gap=26):
    head = (
        f'<div style="font-family:{MONO};font-size:13px;color:{FAINT};'
        f'letter-spacing:0.14em;text-transform:uppercase">{eyebrow}</div>'
        f'<div style="font-family:{DISPLAY};font-weight:700;font-size:{title_size}px;'
        f"line-height:1.08;letter-spacing:-0.025em;color:{INK};max-width:1090px;"
        f'text-wrap:pretty">{title}</div>'
    )
    if sub:
        head += (
            f'<div style="font-size:19px;line-height:1.4;color:{MUTED};'
            f'max-width:960px;text-wrap:pretty">{sub}</div>'
        )
    header = f'<div style="display:flex;flex-direction:column;gap:11px">{head}</div>'
    footer = (
        f'<div style="display:flex;flex-direction:column;gap:10px">'
        f'<div style="height:1px;background:{RULE}"></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'font-family:{MONO};font-size:12px;color:{FAINT};letter-spacing:0.04em">'
        f"<span>{TALK}</span><span>{n:02d} / 18</span></div></div>"
    )
    root = (
        f'<div style="width:1280px;height:720px;box-sizing:border-box;background:{GROUND};'
        f"color:{INK};font-family:{BODY};padding:46px 64px 34px 64px;"
        f'display:flex;flex-direction:column;gap:{gap}px;overflow:hidden">'
        f'{header}<div style="display:flex;flex-direction:column;flex-grow:1;'
        f'min-height:0;gap:20px">{body}</div>{footer}</div>'
    )
    return wrap(root)


def wrap(root):
    return (
        f'<!doctype html>\n<html>\n<head>\n  <meta charset="utf-8">\n'
        f'  <script src="./support.js"></script>\n</head>\n<body>\n<x-dc>\n'
        f'<helmet>\n  <link rel="stylesheet" href="{FONTS}">\n'
        f"  <style>\n    body {{ margin: 0; }}\n"
        f"    a {{ color: {AIFS}; text-decoration: none; }}\n"
        f"    a:hover {{ color: #6C3184; }}\n  </style>\n</helmet>\n"
        f"{root}\n</x-dc>\n</body>\n</html>\n"
    )


def row(inner, gap=24, align="stretch", grow=True):
    g = "flex-grow:1;min-height:0;" if grow else ""
    return f'<div style="display:flex;gap:{gap}px;align-items:{align};{g}">{inner}</div>'


def col(inner, flex=1, gap=16, justify="flex-start"):
    return (
        f'<div style="flex:{flex};display:flex;flex-direction:column;gap:{gap}px;'
        f'justify-content:{justify};min-width:0">{inner}</div>'
    )


def grid(cells, cols=2, gap=20):
    return (
        f'<div style="display:grid;grid-template-columns:repeat({cols},minmax(0,1fr));'
        f'gap:{gap}px;flex-grow:1;min-height:0">{"".join(cells)}</div>'
    )


SL = {}

# ============================ 01  TITLE ======================================
models_strip = "".join(
    f'<div style="display:flex;align-items:center;gap:9px">'
    f'<span style="width:12px;height:12px;background:{c};border-radius:2px"></span>'
    f'<span style="font-family:{MONO};font-size:14px;color:{INK}">{n}</span>'
    f'<span style="font-family:{MONO};font-size:13px;color:{FAINT}">{p_}</span></div>'
    for c, n, p_ in [
        (AURORA, "Aurora", "1.3 B"),
        (GRAPHCAST, "GraphCast", "37 M"),
        (SFNO, "SFNO", "573 M"),
        (AIFS, "AIFS", "255 M"),
    ]
)

SL[1] = wrap(
    f'<div style="width:1280px;height:720px;box-sizing:border-box;background:{GROUND};'
    f"color:{INK};font-family:{BODY};padding:60px 64px 40px 64px;display:flex;"
    f"flex-direction:column;justify-content:space-between;overflow:hidden;"
    f'border-top:8px solid {AIFS}">'
    f'<div style="display:flex;flex-direction:column;gap:26px">'
    f'<div style="font-family:{MONO};font-size:14px;color:{FAINT};letter-spacing:0.16em;'
    f'text-transform:uppercase">Ensembles from a frozen checkpoint</div>'
    f'<div style="font-family:{DISPLAY};font-weight:700;font-size:74px;line-height:1.02;'
    f'letter-spacing:-0.035em;max-width:1050px;text-wrap:pretty">'
    f"Stochastically<br />Perturbed Weights</div>"
    f'<div style="font-size:25px;line-height:1.35;color:{MUTED};max-width:860px;'
    f'text-wrap:pretty">Ensembles from deterministic machine-learning weather models</div>'
    f"</div>"
    f'<div style="display:flex;flex-direction:column;gap:22px">'
    f'<div style="display:flex;gap:32px;flex-wrap:wrap">{models_strip}</div>'
    f'<div style="height:1px;background:{RULE}"></div>'
    f'<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:32px">'
    f'<div style="display:flex;flex-direction:column;gap:7px">'
    f'<div style="font-size:20px;font-weight:600;color:{INK}">'
    f"Simon Adamov &nbsp;&middot;&nbsp; Oliver Fuhrer &nbsp;&middot;&nbsp; "
    f"Reto Knutti &nbsp;&middot;&nbsp; Sebastian Schemm</div>"
    f'<div style="font-size:16px;color:{MUTED}">MeteoSwiss &nbsp;&middot;&nbsp; '
    f"ETH Zurich &nbsp;&middot;&nbsp; University of Cambridge</div></div>"
    f'<div style="font-family:{MONO};font-size:13px;color:{FAINT};text-align:right;'
    f'line-height:1.6">112 initialisations &middot; 10 members &middot; 360 h<br />'
    f"8-way intercomparison</div></div></div></div>"
)

# ============================ 02  THE GAP ====================================
SL[2] = slide(
    2,
    "The gap",
    "Skill without spread",
    row(
        col(
            bullets(
                [
                    "Deterministic MLWMs now match or beat the IFS at 0.25&deg;, at a small "
                    "fraction of the inference cost.",
                    "But a single forecast carries no estimate of its own uncertainty, and the "
                    "atmosphere is chaotic. Operational use needs a distribution.",
                    "The obvious answer is to train a probabilistic model. That costs a dedicated "
                    "training run, and assumes you can retrain at all.",
                ]
            ),
            flex=57,
            gap=18,
        )
        + col(
            f'<div style="background:{WASH};border-left:4px solid {AIFS};padding:26px 26px;'
            f'display:flex;flex-direction:column;gap:14px">'
            + f'<div style="font-family:{DISPLAY};font-weight:600;font-size:25px;line-height:1.24;'
            f'letter-spacing:-0.015em;color:{INK};text-wrap:pretty">'
            f"How much calibrated uncertainty can we recover from a deterministic "
            f"checkpoint that already exists and cannot be retrained?</div>"
            + p("Because the training data, the compute, or the code are out of reach.", 16, FAINT)
            + "</div>"
            + '<div style="display:flex;gap:34px;padding-top:6px">'
            + stat("0", "training runs", AIFS, 42)
            + stat("~60-125 s", "per member", INK, 42)
            + "</div>",
            flex=43,
            gap=20,
            justify="center",
        )
    ),
)


# ============================ 03  FOUR ROUTES ================================
def route(title, body, tag, accent=None):
    return card(
        card_title(title, accent or INK, 21)
        + p(body, 16, MUTED, 1.42)
        + f'<div style="margin-top:auto;font-family:{MONO};font-size:13px;'
        f'color:{accent or FAINT};letter-spacing:0.03em;padding-top:6px">{tag}</div>',
        accent=accent,
        pad=20,
    )


SL[3] = slide(
    3,
    "Related work",
    "Four routes to an ensemble, one of them free",
    grid(
        [
            route(
                "Trained-probabilistic",
                "Learn to generate the ensemble directly, through a "
                "CRPS or diffusion objective. Calibrated by design.",
                "costs 20k-80k GPU-hours of training",
            ),
            route(
                "Initial-condition perturbation",
                "Roll a deterministic net out from perturbed "
                "analyses. Cheap, but a deterministic map sends nearby inputs to nearly "
                "identical outputs.",
                "spread-limited at long lead",
            ),
            route(
                "Deep ensemble",
                "Independently trained checkpoints. The gold standard for " "diversity.",
                "one full training run per member",
            ),
            route(
                "Post-hoc weight perturbation",
                "Generate members by perturbing the weights of "
                "one frozen checkpoint at inference. Training-free, and the spread grows with "
                "lead time.",
                "this work &nbsp;&rarr;&nbsp; SPW",
                AIFS,
            ),
        ],
        cols=2,
        gap=18,
    ),
)

# ============================ 04  THE METHOD =================================
eq = (
    f'<div style="background:{CARD};border:1px solid {RULE2};padding:30px 34px;'
    f'display:flex;flex-direction:column;gap:12px;align-items:center">'
    f'<div style="font-family:{DISPLAY};font-weight:600;font-size:40px;color:{INK};'
    f'letter-spacing:-0.01em">'
    f'<span style="font-style:italic">W&#771;</span><sub style="font-size:22px">m</sub> '
    f'&nbsp;=&nbsp; <span style="font-style:italic">W</span> &nbsp;&#8857;&nbsp; '
    f'( 1 + &sigma; &#958;<sub style="font-size:22px">m</sub> )</div>'
    f'<div style="font-family:{MONO};font-size:15px;color:{FAINT}">'
    f"&#958;<sub>m,i</sub> ~ N(0, 1) i.i.d., one draw per member, per tensor</div></div>"
)

SL[4] = slide(
    4,
    "Method",
    "SPPT perturbs parametrisations. A network has weights.",
    col(
        eq
        + row(
            card(
                card_title("Multiplicative, not additive", INK, 18)
                + p(
                    "Every scalar weight carries variance &sigma;<sup>2</sup>|w|<sup>2</sup>, so "
                    "&sigma; is a dimensionless relative strength and the trained weight "
                    "distribution is preserved. As in Gaussian dropout.",
                    15,
                    MUTED,
                    1.45,
                ),
                pad=18,
            )
            + card(
                card_title("Frozen across the rollout", INK, 18)
                + p(
                    "Member <i>m</i> draws &#958;<sub>m</sub> once and reuses it for every "
                    "autoregressive step. Each member is a fixed weight-space neighbour "
                    "of the checkpoint.",
                    15,
                    MUTED,
                    1.45,
                ),
                pad=18,
            )
            + card(
                card_title("Two degrees of freedom", INK, 18)
                + p(
                    "The selection set <b>S</b> (which tensors to perturb) and the magnitude "
                    "&sigma;. Everything in this talk is a sweep over those two.",
                    15,
                    MUTED,
                    1.45,
                ),
                pad=18,
            ),
            gap=18,
        )
        + p(
            "Weight-space ensembling: rather than averaging many fine-tuned checkpoints as in "
            "model soups, we sample weight-space neighbours of <i>one</i> checkpoint and "
            "ensemble their forecasts.",
            16,
            FAINT,
        ),
        gap=20,
    ),
)


# ============================ 05  THREE QUESTIONS ============================
def question(q, text, sub):
    return (
        f'<div style="display:flex;gap:26px;align-items:flex-start;padding:22px 0;'
        f'border-top:1px solid {RULE}">'
        f'<div style="font-family:{DISPLAY};font-weight:700;font-size:34px;color:{AIFS};'
        f'letter-spacing:-0.02em;width:74px;flex:none">{q}</div>'
        f'<div style="display:flex;flex-direction:column;gap:7px">'
        f'<div style="font-family:{DISPLAY};font-weight:600;font-size:26px;'
        f'letter-spacing:-0.015em;line-height:1.22;color:{INK};text-wrap:pretty">{text}</div>'
        f'<div style="font-family:{MONO};font-size:14px;color:{FAINT};'
        f'letter-spacing:0.03em">{sub}</div></div></div>'
    )


SL[5] = slide(
    5,
    "Three questions",
    "What the paper asks",
    col(
        question(
            "Q1",
            "Does simple weight perturbation work at all, and how far behind "
            "purpose-built probabilistic models does it leave the ensemble?",
            "112-init intercomparison against 3 trained-probabilistic models + IFS-ENS",
        )
        + question(
            "Q2",
            "Where in the network, and on which spatial scales, should the " "noise be injected?",
            "three-phase ablation across four architectures",
        )
        + question(
            "Q3",
            "Where does the scheme fail, and what repairs it?",
            "spread diagnostics, refreshed noise, perturbed initial conditions",
        ),
        gap=0,
        justify="center",
    ),
)

# ============================ 06  SETUP ======================================
SL[6] = slide(
    6,
    "Experimental design",
    "Four backbones, four baselines, two grids",
    row(
        col(
            card_title("Perturbed (frozen public checkpoints)", INK, 17)
            + '<div style="display:flex;flex-direction:column;gap:9px;font-size:16px">'
            + "".join(
                f'<div style="display:flex;justify-content:space-between;gap:12px">'
                f'{chip(c, n)}<span style="font-family:{MONO};font-size:14px;'
                f'color:{FAINT}">{a}</span></div>'
                for c, n, a in [
                    (AURORA, "Aurora", "Swin-3D + Perceiver"),
                    (GRAPHCAST, "GraphCast", "GNN, icosahedral mesh"),
                    (SFNO, "SFNO", "spherical Fourier operator"),
                    (AIFS, "AIFS", "graph enc / transf / graph dec"),
                ]
            )
            + "</div>"
            + f'<div style="height:1px;background:{RULE};margin:6px 0"></div>'
            + card_title("Benchmarked against", INK, 17)
            + '<div style="display:flex;flex-direction:column;gap:9px;font-size:16px">'
            + "".join(
                f'<div style="display:flex;justify-content:space-between;gap:12px">'
                f'{chip(c, n)}<span style="font-family:{MONO};font-size:14px;'
                f'color:{FAINT}">{a}</span></div>'
                for c, n, a in [
                    (AIFSENS, "AIFS-ENS", "trained-probabilistic"),
                    (FCN3, "FourCastNet 3", "trained-probabilistic"),
                    (ATLAS, "Atlas", "trained-probabilistic"),
                    (IFS, "IFS-ENS", "classical reference"),
                ]
            )
            + "</div>",
            flex=48,
            gap=11,
        )
        + col(
            card(
                card_title("Ablation grid", AIFS, 19)
                + p("4 mid-season inits &middot; M = 10 &middot; 240 h", 16, INK)
                + p(
                    "Selects one production baseline per backbone. Selection uses these four "
                    "inits only.",
                    15,
                    MUTED,
                ),
                pad=18,
            )
            + card(
                card_title("Production grid", AIFS, 19)
                + p("112 inits &middot; M = 10 &middot; 360 h", 16, INK)
                + p(
                    "Days 2-8 of Jan / Apr / Jul / Oct, 2023-2024, twice daily. Held out from "
                    "selection: it validates the picks.",
                    15,
                    MUTED,
                ),
                pad=18,
            )
            + p(
                "Scored by one open-source pipeline over 7 variables (2t, msl, z, t, u, v, q at "
                "500 and 850 hPa): CRPSS against the WeatherBench-2 probabilistic climatology, "
                "spread-skill ratio, plus SSIM, LSD, FSS, W<sub>1</sub>, ES, VS and a "
                "signature-kernel score.",
                15,
                FAINT,
            ),
            flex=52,
            gap=14,
        )
    ),
)

# ============================ 07  ABLATION ===================================
SL[7] = slide(
    7,
    "Q2 &middot; design",
    "Where to inject: a three-phase sweep",
    row(
        col(
            "".join(
                f'<div style="display:flex;gap:16px;align-items:flex-start">'
                f'<div style="font-family:{MONO};font-size:13px;color:{AIFS};width:56px;flex:none;'
                f'padding-top:3px;letter-spacing:0.04em">{ph}</div>'
                f'<div style="display:flex;flex-direction:column;gap:5px">'
                f'<div style="font-family:{DISPLAY};font-weight:600;font-size:19px;color:{INK}">{t}</div>'
                f'<div style="font-size:15px;line-height:1.4;color:{MUTED};text-wrap:pretty">{d}</div>'
                f"</div></div>"
                for ph, t, d in [
                    (
                        "PHASE 1",
                        "Magnitude",
                        "All weights, &sigma; &isin; {0.001, 0.003, 0.01, 0.03, 0.1}.",
                    ),
                    (
                        "PHASE 2",
                        "Architectural tensor group",
                        "One group at a time (encoder / processor / decoder), at the variance-budget "
                        "magnitude &sigma;<sub>group</sub> = &sigma;<sub>full</sub> "
                        "&radic;(N<sub>total</sub>/N<sub>group</sub>), so both phases spend the same "
                        "weight-noise budget.",
                    ),
                    (
                        "PHASE 3",
                        "Coarse scales only",
                        "The synoptic-and-larger stage of each backbone: Aurora's U-Net bottleneck, "
                        "GraphCast's 42 coarsest mesh nodes, SFNO's lowest ten spherical-harmonic modes "
                        "(&lambda; &#8819; 3300-5000 km).",
                    ),
                ]
            )
            + f'<div style="height:1px;background:{RULE};margin-top:4px"></div>'
            + p(
                "Selection rule: CRPSS at 240 h, where the deterministic prior has washed out "
                "and the perturbation is doing the work. Rivals within the &plusmn;0.02 grid "
                "scatter must win the full metric bundle. None did.",
                15,
                FAINT,
            ),
            flex=44,
            gap=15,
        )
        + col(
            figure(
                "schematic.png",
                "Phase 1 and 2: tensor groups and their "
                "variance-budget magnitudes, per architecture.",
                maxh=430,
            ),
            flex=56,
        )
    ),
)

# ============================ 08  Q1 RESULT ==================================
crpss_rows = [
    ((chip(AIFSENS, "AIFS-ENS"), "0.256", "ref"), {"weight": 600}),
    ((chip(ATLAS, "Atlas"), "0.238", "0.018"), {}),
    ((chip(FCN3, "FourCastNet 3"), "0.218", "0.038"), {}),
    ((chip(IFS, "IFS-ENS"), "0.207", "0.049"), {}),
    ((chip(AIFS, "AIFS + SPW"), "0.221", "0.036"), {"weight": 600, "bg": WASH}),
    ((chip(GRAPHCAST, "GraphCast + SPW"), "0.167", "0.089"), {"weight": 600, "bg": WASH}),
    ((chip(AURORA, "Aurora + SPW"), "0.134", "0.122"), {"weight": 600, "bg": WASH}),
    ((chip(SFNO, "SFNO + SPW"), "0.125", "0.131"), {"weight": 600, "bg": WASH}),
]
SL[8] = slide(
    8,
    "Q1 &middot; result",
    "It works: 0.04 to 0.13 CRPSS behind the best trained model",
    row(
        col(
            figure(
                "crpss.png",
                "CRPSS, 6-360 h, variable mean over the 7 paper variables. "
                "Solid: SPW. Dashed: trained-probabilistic. Dotted: IFS-ENS.",
                maxh=402,
            ),
            flex=41,
        )
        + col(
            table(["CRPSS at 240 h", "value", "gap"], crpss_rows, [52, 22, 22], size=15, rowpad=6)
            + p(
                "Paired block bootstrap over the eight weekly blocks confirms every gap: "
                "0.036 [0.031, 0.041] for AIFS up to 0.131 [0.121, 0.141] for SFNO.",
                14,
                FAINT,
            )
            + f'<div style="background:{WASH};border-left:4px solid {AIFS};padding:16px 18px">'
            + p(
                "Perturbed AIFS also beats the operational IFS-ENS, at zero training cost.", 16, INK
            )
            + "</div>",
            flex=59,
            gap=14,
        )
    ),
)


# ============================ 09  COST =======================================
def bar(label, colour, secs, maxs=2280, note=""):
    w = max(1.4, 100 * (secs / maxs) ** 0.42)
    return (
        f'<div style="display:flex;align-items:center;gap:14px">'
        f'<div style="width:180px;flex:none;font-size:16px">{chip(colour, label)}</div>'
        f'<div style="flex:1;height:20px;background:{RULE2};position:relative">'
        f'<div style="width:{w:.1f}%;height:100%;background:{colour}"></div></div>'
        f'<div style="width:150px;flex:none;text-align:right;font-family:{MONO};'
        f'font-size:15px;color:{INK}">{secs} s{note}</div></div>'
    )


SL[9] = slide(
    9,
    "Cost",
    "Nothing to train, seconds to run",
    col(
        '<div style="display:flex;flex-direction:column;gap:11px">'
        + bar("SFNO + SPW", SFNO, 59)
        + bar("AIFS + SPW", AIFS, 75)
        + bar("AIFS-ENS", AIFSENS, 75)
        + bar("Aurora + SPW", AURORA, 90)
        + bar("FourCastNet 3", FCN3, 114)
        + bar("GraphCast + SPW", GRAPHCAST, 125)
        + bar("Atlas", ATLAS, 2280, note=" &nbsp;(38 min)")
        + "</div>"
        + p(
            "Median pure-inference seconds per member for a 360 h forecast, NVIDIA GH200. "
            "Atlas is 18-39&times; slower than the other six: its multi-step SDE sampler.",
            15,
            FAINT,
        )
        + f'<div style="height:1px;background:{RULE}"></div>'
        + row(
            stat("0", "SPW training cost", AIFS, 40)
            + stat("~20,000", "GPU-h &middot; AIFS-ENS", AIFSENS, 40)
            + stat("~80,000", "GPU-h &middot; FCN3", FCN3, 40)
            + stat("undisclosed", "Atlas", ATLAS, 40),
            gap=40,
            grow=False,
        ),
        gap=18,
        justify="center",
    ),
)


# ============================ 10  Q2 RESULT ==================================
def winner(colour, model, site, sigma, why):
    return card(
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f"{card_title(model, colour, 22)}"
        f'<span style="font-family:{MONO};font-size:15px;color:{FAINT}">'
        f"&sigma; = {sigma}</span></div>"
        + f'<div style="font-family:{DISPLAY};font-weight:600;font-size:19px;'
        f'color:{INK};letter-spacing:-0.01em">{site}</div>' + p(why, 15, MUTED, 1.4),
        pad=18,
    )


SL[10] = slide(
    10,
    "Q2 &middot; result",
    "No injection site works across models",
    col(
        grid(
            [
                winner(
                    AURORA,
                    "Aurora",
                    "Encoder",
                    "0.025",
                    "Phase 2b. The stage that produces the Perceiver latent.",
                ),
                winner(
                    GRAPHCAST,
                    "GraphCast",
                    "All weights",
                    "0.01",
                    "Phase 1. Its weight-shared mesh localises no subset by scale.",
                ),
                winner(
                    SFNO,
                    "SFNO",
                    "Spectral modes &#8467; &lt; 10",
                    "0.25",
                    "Phase 3. A literal coarse-scale cut, &lambda; &#8819; 4000 km.",
                ),
                winner(
                    AIFS,
                    "AIFS",
                    "Decoder",
                    "0.028",
                    "Phase 2. The stage that reads the O96 processor mesh.",
                ),
            ],
            cols=4,
            gap=15,
        )
        + f'<div style="background:{WASH};border-left:4px solid {INK};padding:18px 22px">'
        + f'<div style="font-family:{DISPLAY};font-weight:600;font-size:23px;line-height:1.28;'
        f'letter-spacing:-0.015em;color:{INK};text-wrap:pretty">Four winners, four different '
        f"places in four different architectures. SPW is a tuning procedure, not a "
        f"plug-and-play recipe: each new checkpoint costs a sweep.</div></div>",
        gap=18,
    ),
)

# ============================ 11  Q2 INTERPRETATION ==========================
SL[11] = slide(
    11,
    "Q2 &middot; interpretation",
    "What the winners share: proximity to the low-resolution latent",
    row(
        col(
            bullets(
                [
                    "Aurora's <b>encoder</b> produces the Perceiver latent. AIFS's <b>decoder</b> reads "
                    "the O96 processor mesh. SFNO's <b>&#8467; &lt; 10</b> is the coarse scale by "
                    "construction. Each borders the low-resolution stage rather than the "
                    "full-resolution interior.",
                    "Perturbing the <b>interior</b> underdisperses: Aurora's backbone and the central "
                    "processors of AIFS and SFNO carry normalisation layers that damp the injected "
                    "variance.",
                    "Not the trained site: AIFS-ENS injects its learned noise in exactly the processor "
                    "we cannot exploit.",
                    "GraphCast falls outside the pattern entirely.",
                ],
                size=17,
                gap=15,
            ),
            flex=58,
        )
        + col(
            card(
                card_title("The honest caveat", INK, 19)
                + p(
                    "Four architectures cannot establish a rule. Independent latent-space work "
                    "finds the same coarse-to-fine organisation in GraphCast and Aurora, which "
                    "makes this a hypothesis worth testing on more models.",
                    15,
                    MUTED,
                ),
                pad=18,
            )
            + card(
                card_title("What to do on an unseen checkpoint", AIFS, 19)
                + p(
                    "Start with <b>all weights at &sigma; = 0.01 to 0.03</b>, then move the "
                    "perturbation onto the stage bordering the low-resolution latent, where "
                    "one exists.",
                    15,
                    MUTED,
                ),
                accent=AIFS,
                pad=18,
            ),
            flex=42,
            gap=16,
        )
    ),
)

# ============================ 12  Q3 FAILURE =================================
SL[12] = slide(
    12,
    "Q3 &middot; failure",
    "Calibrated pointwise, overdispersed on the domain mean",
    row(
        col(
            row(
                card(
                    f'<div style="font-family:{MONO};font-size:13px;color:{FAINT};'
                    f'letter-spacing:0.06em;text-transform:uppercase">Per-pixel SSR, 240 h</div>'
                    + f'<div style="font-family:{DISPLAY};font-weight:700;font-size:40px;'
                    f'color:{INK};letter-spacing:-0.02em">0.94 - 1.11</div>'
                    + p("All four baselines. Reliable.", 15, MUTED),
                    pad=18,
                )
                + card(
                    f'<div style="font-family:{MONO};font-size:13px;color:{FAINT};'
                    f'letter-spacing:0.06em;text-transform:uppercase">Spatial-mean SSR, 240 h</div>'
                    + '<div style="display:flex;flex-direction:column;gap:5px;padding-top:2px">'
                    + "".join(
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-size:16px">{chip(c, n)}'
                        f'<span style="font-family:{MONO};font-weight:500;color:{INK}">{v}</span>'
                        f"</div>"
                        for c, n, v in [
                            (GRAPHCAST, "GraphCast", "3.2"),
                            (AIFS, "AIFS", "2.5"),
                            (AURORA, "Aurora", "1.6"),
                            (SFNO, "SFNO", "0.66"),
                        ]
                    )
                    + "</div>"
                    + p("Trained-probabilistic band: 0.62 - 1.42.", 14, FAINT),
                    accent=ATLAS,
                    pad=18,
                ),
                gap=16,
            )
            + f'<div style="background:{WASH};border-left:4px solid {ATLAS};padding:18px 22px">'
            + f'<div style="font-size:18px;line-height:1.42;color:{INK};text-wrap:pretty">'
            f"<b>The mechanism.</b> Each member runs the whole forecast through one perturbed "
            f"weight set, so the entire field shifts together. Per-pixel spread averages many "
            f"local perturbation directions and stays calibrated; the domain mean does not."
            f"</div></div>"
            + p(
                "It is a property of the perturbation target, not the model: SFNO's own "
                "all-weights and encoder/decoder cells overdisperse the same way (SSR 3.0, 1.8, "
                "2.4). Only the coarse-mode restriction holds the ratio at or below 1.",
                15,
                FAINT,
            ),
            flex=53,
            gap=15,
        )
        + col(
            figure("ssr.png", "Spatial-mean SSR against lead time, 2 of 7 variables.", maxh=310),
            flex=47,
            justify="center",
        )
    ),
)

# ============================ 13  REPAIR 1 ===================================
SL[13] = slide(
    13,
    "Q3 &middot; repair 1",
    "Refresh the noise across the rollout",
    row(
        col(
            f'<div style="background:{CARD};border:1px solid {RULE2};padding:18px 22px;'
            f'display:flex;align-items:center;justify-content:center;gap:16px">'
            f'<span style="font-family:{DISPLAY};font-weight:600;font-size:28px;color:{INK}">'
            f'&sigma;<sub style="font-size:17px">N</sub> = '
            f'&sigma;<sub style="font-size:17px">frozen</sub> &radic;(T / N)</span></div>'
            + p(
                "Hold the draw fixed for N steps, resample at the boundary, and boost the "
                "per-draw magnitude so the accumulated weight perturbation matches the frozen "
                "case in variance. The coherent drift becomes a partly-cancelling random walk.",
                16,
                MUTED,
            )
            + '<div style="display:flex;flex-direction:column;gap:10px">'
            + "".join(
                f'<div style="display:flex;gap:14px;align-items:baseline;font-size:16px">'
                f'<span style="width:150px;flex:none">{chip(c, n)}</span>'
                f'<span style="font-family:{MONO};font-size:15px;color:{INK}">{v}</span>'
                f'<span style="font-size:15px;color:{MUTED}">{w}</span></div>'
                for c, n, v, w in [
                    (SFNO, "SFNO", "0.66 &rarr; 0.83", "works, at no CRPS cost"),
                    (AURORA, "Aurora", "1.56 &rarr; 1.43", "barely moves, costs 1-5% CRPS"),
                    (AIFS, "AIFS", "2.54 &rarr; 2.50", "unchanged"),
                    (GRAPHCAST, "GraphCast", "0.27 &rarr; 0.15", "inverts"),
                ]
            )
            + "</div>",
            flex=52,
            gap=16,
        )
        + col(
            figure("refresh.png", "Spatial-mean SSR, frozen vs refresh-every-20.", maxh=248)
            + f'<div style="background:{WASH};border-left:4px solid {GRAPHCAST};'
            f'padding:16px 20px">'
            + p(
                "<b>GraphCast inverts the sign.</b> Its weight-shared mesh forces the "
                "coarse perturbation into activation space, where per-step noise damps "
                "rather than accumulating.",
                15,
                INK,
            )
            + "</div>"
            + p("A targeted lever for the scale-localised regime, not a general fix.", 15, FAINT),
            flex=48,
            gap=14,
        )
    ),
)

# ============================ 14  REPAIR 2 ===================================
SL[14] = slide(
    14,
    "Q3 &middot; repair 2",
    "Perturbed initial conditions carry the early lead",
    row(
        col(
            p(
                "Keep each backbone's production weight perturbation, and initialise member "
                "<i>m</i> from the <i>m</i>-th perturbed analysis of the IFS-ENS ensemble of "
                "data assimilations. A physically structured ensemble, not synthetic noise: "
                "small unstructured input perturbations are known to under-disperse ML "
                "ensembles.",
                17,
                MUTED,
            )
            + card(
                card_title("The two sources combine in variance, and swap roles", INK, 18)
                + '<div style="display:flex;gap:34px;padding-top:4px">'
                + stat("8-17%", "weight share at 6 h", MUTED, 38)
                + stat("86-97%", "weight share at 240 h", AIFS, 38)
                + "</div>"
                + p(
                    "&sigma;<sup>2</sup><sub>wt+ic</sub> matches &sigma;<sup>2</sup><sub>wt</sub> "
                    "+ &sigma;<sup>2</sup><sub>ic</sub> to within 1% at 6 h, falling to "
                    "0.59-0.69 of it by 240 h as both saturate toward climatology.",
                    15,
                    MUTED,
                ),
                pad=18,
            ),
            flex=52,
            gap=16,
        )
        + col(
            card_title("What it buys", INK, 19)
            + bullets(
                [
                    "24 h per-pixel MSL SSR rises by +0.09 to +0.20 across backbones (AIFS: "
                    "0.65 &rarr; 0.79).",
                    "It partly breaks the frozen whole-field shift: GraphCast's spatial-mean SSR "
                    "falls 3.20 &rarr; 1.90, the practical fix for the one backbone whose "
                    "architecture excludes the weight-space refresh.",
                    "Variable-mean CRPSS stays within &plusmn;0.02 of weight-only from 24 h on, "
                    "at a 0.06-0.10 cost at 6 h, where the injected analysis spread is not yet "
                    "skill-bearing.",
                    "On the 240 h signature kernel, weight+IC is the best arm for every backbone "
                    "and IC-only the worst: sub-additive in variance, super-additive in path "
                    "realism.",
                ],
                size=16,
                gap=12,
            ),
            flex=48,
            gap=14,
        )
    ),
)

# ============================ 15  MILTON =====================================
milton_rows = [
    ((chip(ATLAS, "Atlas"), "89", "281"), {}),
    ((chip(FCN3, "FourCastNet 3"), "87", "438"), {}),
    ((chip(AIFSENS, "AIFS-ENS"), "86", "300"), {}),
    ((chip(IFS, "IFS-ENS"), "71", "317"), {}),
    ((chip(SFNO, "SFNO + IC"), "76", "306"), {"bg": WASH, "weight": 600}),
    ((chip(GRAPHCAST, "GraphCast + IC"), "73", "258"), {"bg": WASH, "weight": 600}),
    ((chip(AIFS, "AIFS + IC"), "67", "262"), {"bg": WASH, "weight": 600}),
    ((chip(AURORA, "Aurora + IC"), "66", "<b>218</b>"), {"bg": WASH, "weight": 600}),
]
SL[15] = slide(
    15,
    "Case study",
    "Hurricane Milton, October 2024",
    row(
        col(
            p(
                "14 initialisations, 2 to 8 October. Tracked with TempestExtremes against "
                "IBTrACS v4, identically for all eight baselines. The SPW rows use the "
                "IC-augmented arm, since weight-only is thin exactly where an RI case probes.",
                15,
                MUTED,
            )
            + table(
                ["Baseline", "detect %", "pos. err km"],
                milton_rows,
                [50, 22, 26],
                size=15,
                rowpad=5,
            )
            + f'<div style="background:{WASH};border-left:4px solid {AURORA};padding:14px 18px">'
            + p(
                "The ordering inverts between the two columns: the trained models detect the "
                "storm more often, three of the four SPW ensembles place it better.",
                15,
                INK,
            )
            + "</div>",
            flex=47,
            gap=13,
        )
        + col(
            figure(
                "milton.png",
                "Member tracks, two initialisations. Black: IBTrACS best "
                "track. Detected members in parentheses.",
                maxh=400,
            ),
            flex=53,
        )
    ),
)

# ============================ 16  WHAT IC BUYS ===============================
SL[16] = slide(
    16,
    "Case study",
    "On a rapid-intensification case, the spread is the analysis",
    row(
        col(
            figure("icspread.png", "AIFS, three arms plus two reference ensembles.", maxh=270)
            + row(
                stat("48 &rarr; 96 km", "early-lead track cone", AIFS, 32)
                + stat("1.1 &rarr; 1.9 hPa", "intensity spread", AIFS, 32),
                gap=30,
                grow=False,
            ),
            flex=56,
            gap=16,
        )
        + col(
            bullets(
                [
                    "IC augmentation barely moves the ensemble mean and sharply changes the "
                    "dispersion. The IC-only curve stays flat while weight-only grows sixfold: the "
                    "two arms carry different lead-time ranges.",
                    "Every baseline's MSL bias sits at +37 to +44 hPa. That is the verification "
                    "chain, not skill: the 0.25&deg; ERA5 grid bottoms out at 976 hPa against the "
                    "895 hPa IBTrACS peak.",
                    "One event is not decisive, but variable-mean CRPSS does not obviously mis-rank "
                    "a high-impact extreme.",
                ],
                size=16,
                gap=13,
            ),
            flex=44,
        )
    ),
)

# ============================ 17  LIMITATIONS ================================
SL[17] = slide(
    17,
    "Limitations",
    "What this does not show",
    grid(
        [
            card(
                card_title("Sweep cost", INK, 18)
                + p(
                    "Every ablation cell is a full 10-member run. The per-checkpoint "
                    "configuration search is the price paid for the absent training run.",
                    15,
                    MUTED,
                ),
                pad=17,
            ),
            card(
                card_title("No within-family baseline", INK, 18)
                + p(
                    "MC-dropout and SWAG need training access the frozen public checkpoints "
                    "lack, so a fair intra-family comparison is out of reach.",
                    15,
                    MUTED,
                ),
                pad=17,
            ),
            card(
                card_title("Sampling uncertainty", INK, 18)
                + p(
                    "The 112 inits are twice-daily within eight month-blocks, so the effective "
                    "sample size is closer to eight than to 112.",
                    15,
                    MUTED,
                ),
                pad=17,
            ),
            card(
                card_title("M = 10, not 50", INK, 18)
                + p(
                    "Enough to support the ranking, but the 10-member SSR and CRPSS remain "
                    "biased in absolute terms.",
                    15,
                    MUTED,
                ),
                pad=17,
            ),
            card(
                card_title("An external IC ensemble", INK, 18)
                + p(
                    "The IC augmentation draws from the IFS-ENS EDA. Where none is available, "
                    "cruder substitutes need not transfer the calibration.",
                    15,
                    MUTED,
                ),
                pad=17,
            ),
            card(
                card_title("No precipitation", ATLAS, 18)
                + p(
                    "Humidity, the most non-Gaussian field we do score, is already the worst "
                    "calibrated. Perturbing a smoothed solution adds spread but cannot restore "
                    "sharpness the parent never produced.",
                    15,
                    MUTED,
                ),
                accent=ATLAS,
                pad=17,
            ),
        ],
        cols=3,
        gap=16,
    ),
)


# ============================ 18  TAKEAWAYS ==================================
def takeaway(n, text):
    return (
        f'<div style="display:flex;gap:20px;align-items:flex-start;padding:15px 0;'
        f'border-top:1px solid {RULE}">'
        f'<div style="font-family:{MONO};font-size:15px;color:{AIFS};width:28px;'
        f'flex:none;padding-top:5px">{n}</div>'
        f'<div style="font-size:19px;line-height:1.4;color:{INK};text-wrap:pretty">'
        f"{text}</div></div>"
    )


SL[18] = slide(
    18,
    "Conclusions",
    "Take-aways",
    col(
        takeaway(
            "01",
            "A frozen deterministic checkpoint already carries a usable ensemble: "
            "<b>0.04 to 0.13 CRPSS</b> behind the best trained model at 240 h, at zero "
            "marginal training cost.",
        )
        + takeaway(
            "02",
            "The injection site is <b>architecture-specific</b>. Budget a sweep "
            "per checkpoint; start at all weights, &sigma; = 0.01 to 0.03.",
        )
        + takeaway(
            "03",
            "The failure is the <b>domain mean</b>, and it belongs to the "
            "perturbation target: scale-restrict the noise, or add perturbed initial "
            "conditions.",
        )
        + takeaway(
            "04",
            "Which ensemble for which job: trained-probabilistic where it covers "
            "your variables, SPW for the checkpoints and centres it does not.",
        )
        + f'<div style="height:1px;background:{RULE};margin-top:2px"></div>'
        + row(
            col(
                p(
                    "<b>Next:</b> choose the injection site by Fisher information instead of "
                    "searching it; steer the draw rather than randomise it; carry the "
                    "framework to limited-area models.",
                    16,
                    MUTED,
                ),
                flex=60,
            )
            + col(
                f'<div style="font-family:{MONO};font-size:14px;color:{FAINT};'
                f'text-align:right;line-height:1.7">'
                f"github.com/MeteoSwiss/ai-models-ensembles &middot; v1.0.0<br />"
                f"simon.adamov@env.ethz.ch</div>",
                flex=40,
            ),
            gap=24,
            grow=False,
        ),
        gap=0,
        justify="center",
    ),
)

# --- write -------------------------------------------------------------------
NAMES = {1: "Main"}
for i in range(2, 19):
    NAMES[i] = f"Slide{i:02d}"

TITLES = {
    1: "01 Title",
    2: "02 The gap",
    3: "03 Four routes",
    4: "04 Method: SPW",
    5: "05 Three questions",
    6: "06 Experimental design",
    7: "07 Three-phase ablation",
    8: "08 Q1 result: CRPSS",
    9: "09 Cost",
    10: "10 Q2 result: four winners",
    11: "11 Q2 interpretation",
    12: "12 Q3 failure: domain mean",
    13: "13 Repair 1: refresh",
    14: "14 Repair 2: perturbed ICs",
    15: "15 Milton",
    16: "16 Milton: what ICs buy",
    17: "17 Limitations",
    18: "18 Take-aways",
}

NOTES = {
    1: (
        "0:00 - 0:25",
        "Title, 25 s. Name, affiliations, one line: we turn deterministic AI "
        "weather models into ensembles by perturbing their weights at inference. Four backbones, "
        "112 initialisations. Go straight to slide 2.",
    ),
    2: (
        "0:25 - 1:25",
        "60 s. Set the gap in three beats: MLWMs are as good as the IFS and far "
        "cheaper; they give one forecast; retraining is often not an option. Land the question "
        "on the right verbatim - it is the paper's thesis. Do not explain CRPS yet.",
    ),
    3: (
        "1:25 - 2:30",
        "65 s. Fast. The 2x2 exists to justify one choice: post-hoc is the only "
        "route that is both training-free and gives spread that grows with lead time. IC "
        "perturbation comes back later as a complement, not a rival.",
    ),
    4: (
        "2:30 - 3:50",
        "80 s. The whole method is one equation. Emphasise multiplicative: sigma "
        "is dimensionless, so the same number means the same thing in a 37 M and a 1.3 B model. "
        "Say 'weight-space neighbours of one checkpoint'. Two knobs: which tensors, how hard.",
    ),
    5: (
        "3:50 - 4:25",
        "35 s. Read the three questions and say the talk answers them in order. "
        "This is the roadmap slide - do not linger.",
    ),
    6: (
        "4:25 - 5:25",
        "60 s. Point at the two grids: selection happens on four inits, the "
        "112-init grid is held out and validates the picks. That design decision pre-empts the "
        "obvious overfitting question. Mention the pipeline is open source.",
    ),
    7: (
        "5:25 - 6:45",
        "80 s. Walk the three phases, then the variance budget: sigma_group scales "
        "as sqrt(N_total/N_group) so a small group gets a larger sigma and every phase spends the "
        "same noise budget. Mention the +/-0.02 scatter and the tie-break rule.",
    ),
    8: (
        "6:45 - 8:20",
        "95 s. The Q1 answer. Give the four gaps, then the two things that matter: "
        "zero training cost, and perturbed AIFS beats the operational IFS-ENS. Flag that the 360 h "
        "gaps narrow only because everything decays to climatology. Bootstrap CIs if pressed.",
    ),
    9: (
        "8:20 - 9:05",
        "45 s. Quick. Six models within 59-125 s per member, Atlas 38 min because "
        "of its SDE sampler. Then the training row: this is where SPW actually wins.",
    ),
    10: (
        "9:05 - 10:25",
        "80 s. The central negative result and the honest one. Four winners, four "
        "different places. Say plainly: this is a tuning procedure, not plug-and-play. Expect a "
        "question here - the answer is slide 11.",
    ),
    11: (
        "10:25 - 11:35",
        "70 s. The interpretive claim, carefully hedged. Winners border the "
        "low-resolution latent; interiors have normalisation layers that damp the variance; "
        "GraphCast does not fit. Four models cannot establish a rule - say that out loud. End on "
        "the practical recipe.",
    ),
    12: (
        "11:35 - 13:10",
        "95 s. The most important failure slide. Two numbers side by side: "
        "pointwise fine, domain mean broken. Then the mechanism in one sentence - one weight set "
        "per member shifts the whole field together. The SFNO control proves it is the "
        "perturbation target, not the model.",
    ),
    13: (
        "13:10 - 14:25",
        "75 s. The variance-budget rule keeps the comparison fair. It works for "
        "SFNO and essentially nowhere else, and GraphCast flips sign because its noise lives in "
        "activations, not weights. Report the negative result plainly.",
    ),
    14: (
        "14:25 - 15:40",
        "75 s. The complementary lever. The variance decomposition is the "
        "cleanest result on the slide: weights supply 8-17% of the spread at 6 h and 86-97% at "
        "240 h. That is why the headline uses weight-only and Milton uses weight+IC.",
    ),
    15: (
        "15:40 - 17:00",
        "80 s. Set the case up in two sentences, then read the table inversion: "
        "trained models detect more often, SPW places better, Aurora+IC is the best position "
        "error of all eight. Let the track figure carry the rest.",
    ),
    16: (
        "17:00 - 18:10",
        "70 s. IC augmentation doubles the early track cone and widens intensity "
        "spread ~70%. Pre-empt the MSL bias question: +37 to +44 hPa for everyone is the 0.25 deg "
        "grid, not the forecast.",
    ),
    17: (
        "18:10 - 19:05",
        "55 s. Do not read all six. Say the three that reviewers ask about: "
        "sweep cost, eight effective blocks not 112, and no precipitation. Then move on.",
    ),
    18: (
        "19:05 - 20:00",
        "55 s. Four sentences, one per line, then stop. Leave the repo link on "
        "screen for questions. Backup answers: MC-dropout/SWAG need training access; M=10 is "
        "defensible per Leutbecher (2019); GraphCast is not bit-reproducible because it runs in "
        "JAX/XLA.",
    ),
}

# canvas layout: 3 columns x 6 rows, sticky note to the right of each slide
SW, SH, NW = 1280, 720, 300
COL_PITCH, ROW_PITCH = SW + 30 + NW + 90, SH + 150
artboards, annotations = [], []
for i in range(1, 19):
    c, r = (i - 1) % 3, (i - 1) // 3
    x, y = c * COL_PITCH, r * ROW_PITCH
    fname = f"{NAMES[i]}.dc.html"
    (HERE / fname).write_text(SL[i], encoding="utf-8")
    artboards.append({"file": fname, "x": x, "y": y, "w": SW, "h": SH, "title": TITLES[i]})
    clock, note = NOTES[i]
    annotations.append(
        {"id": f"notes-{i:02d}", "x": x + SW + 30, "y": y, "w": NW, "text": f"{clock}\n\n{note}"}
    )

(HERE / "canvas.json").write_text(
    json.dumps(
        {"artboards": artboards, "annotations": annotations, "launch": {"view": "canvas"}}, indent=2
    ),
    encoding="utf-8",
)
print(f"wrote {len(artboards)} artboards + canvas.json")
