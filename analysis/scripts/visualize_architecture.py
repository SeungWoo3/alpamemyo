"""
visualize_architecture.py

Architecture block diagram for Alpamayo-R1-10B.

Component summary
-----------------
Vision Encoder : PatchEmbed (patch=16) + 27 ViT blocks (hidden=1152) + PatchMerger (→4096)
VLM            : Qwen3-VL-8B  hidden=4096, 36 Transformer layers, GQA
Expert Decoder : hidden=2048, 16 heads (cross-attention conditioning)
Diffusion      : 10 Euler steps, Flow Matching
Action Space   : 64 waypoints × 2 dims  →  Unicycle Model  →  (64,3) trajectory
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
})

# ── Colour palette (one per major pipeline stage) ───────────────────────────
COL = {
    "input"   : "#B3E5FC",   # light blue
    "vision"  : "#CE93D8",   # light purple
    "text"    : "#80CBC4",   # teal
    "vlm"     : "#FFB74D",   # amber
    "expert"  : "#EF9A9A",   # light red
    "diffusion": "#A5D6A7",  # light green
    "action"  : "#FFF176",   # yellow
    "traj"    : "#F48FB1",   # pink
    "arrow"   : "#455A64",   # dark blue-grey
    "bg"      : "#FAFAFA",
    "panel"   : "#ECEFF1",
}

def _darken(hex_color: str, amount: float = 0.3) -> str:
    """Return a darkened version of a hex colour."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    r = int(r * (1 - amount))
    g = int(g * (1 - amount))
    b = int(b * (1 - amount))
    return f"#{r:02x}{g:02x}{b:02x}"

EDGE_COL = {k: _darken(v, 0.35) for k, v in COL.items()
            if k not in ("arrow", "bg", "panel")}


# ── Drawing primitives ──────────────────────────────────────────────────────

def draw_block(ax, x, y, w, h,
               label: str, sublabel: str = "",
               color: str = "#BBDEFB", edge_color: str = "#1565C0",
               fontsize: int = 10, sublabel_fontsize: int = 8.5,
               radius: float = 0.12, zorder: int = 3):
    """Draw a rounded rectangle with a two-line label."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor=edge_color,
        linewidth=1.6, zorder=zorder,
    )
    ax.add_patch(box)

    cy = y if not sublabel else y + h * 0.14
    ax.text(x, cy, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            zorder=zorder + 1)
    if sublabel:
        ax.text(x, y - h * 0.22, sublabel,
                ha="center", va="center",
                fontsize=sublabel_fontsize, color="#424242",
                zorder=zorder + 1)


def draw_arrow(ax, x0, y0, x1, y1,
               label: str = "", label_side: str = "right",
               color: str = COL["arrow"], lw: float = 1.8,
               arrowstyle: str = "-|>", mutation_scale: float = 16):
    """Draw a directed arrow between two points."""
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color, lw=lw,
            mutation_scale=mutation_scale,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        offset = (0.18, 0) if label_side == "right" else (-0.18, 0)
        if abs(y1 - y0) < 0.1:   # horizontal arrow → label above
            offset = (0, 0.18)
        ax.text(mx + offset[0], my + offset[1], label,
                ha="center", va="center",
                fontsize=8, color="#37474F",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))


def draw_bracket_group(ax, x_left, x_right, y_top, y_bot,
                        label: str, color: str = "#90A4AE"):
    """Draw a vertical bracket grouping several blocks."""
    pad = 0.15
    ax.annotate(
        "", xy=(x_left - pad, y_bot), xytext=(x_left - pad, y_top),
        arrowprops=dict(arrowstyle="-", color=color, lw=1.2,
                        connectionstyle="arc3,rad=0"),
    )
    ax.text(x_left - pad - 0.08, (y_top + y_bot) / 2, label,
            ha="right", va="center", fontsize=8, color=color,
            rotation=90, fontweight="bold")


# ── Layout constants ────────────────────────────────────────────────────────
#
#  The diagram is laid out on a [0, 18] × [0, 12] canvas.
#  Main data flow runs left → right across three rows:
#
#  Row A (y≈9.0): Image Input → Vision Encoder sub-blocks → Vision Tokens
#  Row B (y≈5.5): Text Input  → Tokenizer → Text Tokens
#  Row C (y≈7.5): [Vision+Text Tokens] → VLM → Expert Tokens
#  Row D (y≈3.5): Noise → Diffusion Loop (with Expert conditioning)
#  Row E (y≈1.5): Actions → Unicycle → Trajectory
#

def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor(COL["bg"])
    ax.set_facecolor(COL["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(
        9, 11.6,
        "Alpamayo-R1-10B  —  Architecture Overview",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color="#1A237E",
    )
    ax.text(
        9, 11.2,
        "Vision Encoder (27 ViT layers, hidden=1152)  ·  VLM Qwen3-VL-8B (36 layers, GQA)  ·  "
        "Expert Decoder (hidden=2048)  ·  Flow Matching Diffusion (10 steps)",
        ha="center", va="center",
        fontsize=9.5, color="#546E7A",
    )

    # ══════════════════════════════════════════════════════════════════════
    # ROW A — Vision pipeline  (y = 9.2)
    # ══════════════════════════════════════════════════════════════════════
    Y_VIS = 9.2

    # Image Input
    draw_block(ax, 1.0, Y_VIS, 1.3, 0.75,
               "Image Input", "H × W × 3",
               color=COL["input"], edge_color=_darken(COL["input"]))

    # PatchEmbed
    draw_block(ax, 2.9, Y_VIS, 1.5, 0.75,
               "PatchEmbed",  "patch=16\n1152-dim",
               color=COL["vision"], edge_color=_darken(COL["vision"]))

    # 27 ViT blocks
    draw_block(ax, 5.1, Y_VIS, 2.2, 0.90,
               "27 × ViT Block",
               "hidden=1152, MHA\n~427M params",
               color=COL["vision"], edge_color=_darken(COL["vision"]))

    # PatchMerger
    draw_block(ax, 7.5, Y_VIS, 1.6, 0.75,
               "PatchMerger",  "1152 → 4096",
               color=COL["vision"], edge_color=_darken(COL["vision"]))

    # Vision Tokens
    draw_block(ax, 9.5, Y_VIS, 1.5, 0.75,
               "Vision Tokens", "seq_len × 4096",
               color=COL["input"], edge_color=_darken(COL["input"]))

    # Arrows — vision pipeline
    draw_arrow(ax, 1.65, Y_VIS, 2.15, Y_VIS)
    draw_arrow(ax, 3.65, Y_VIS, 4.0,  Y_VIS)
    draw_arrow(ax, 6.2,  Y_VIS, 6.7,  Y_VIS)
    draw_arrow(ax, 8.3,  Y_VIS, 8.75, Y_VIS)

    # Vision Encoder brace label
    ax.annotate("", xy=(8.4, Y_VIS + 0.62), xytext=(2.1, Y_VIS + 0.62),
                arrowprops=dict(arrowstyle="-", color=_darken(COL["vision"]),
                                lw=1.0, linestyle="dashed"))
    ax.text(5.25, Y_VIS + 0.76,
            "Vision Encoder  (SigLIP-style)",
            ha="center", va="center", fontsize=8.5,
            color=_darken(COL["vision"]), fontstyle="italic")

    # ══════════════════════════════════════════════════════════════════════
    # ROW B — Text pipeline  (y = 5.8)
    # ══════════════════════════════════════════════════════════════════════
    Y_TXT = 5.8

    draw_block(ax, 1.0, Y_TXT, 1.3, 0.75,
               "Text Input", "Tokens",
               color=COL["input"], edge_color=_darken(COL["input"]))

    draw_block(ax, 2.9, Y_TXT, 1.5, 0.75,
               "Tokenizer", "BPE / WordPiece",
               color=COL["text"], edge_color=_darken(COL["text"]))

    draw_block(ax, 5.1, Y_TXT, 1.6, 0.75,
               "Text Tokens", "seq_len × 4096",
               color=COL["input"], edge_color=_darken(COL["input"]))

    draw_arrow(ax, 1.65, Y_TXT, 2.15, Y_TXT)
    draw_arrow(ax, 3.65, Y_TXT, 4.3,  Y_TXT)

    # ══════════════════════════════════════════════════════════════════════
    # Merge arrows — Vision Tokens + Text Tokens → VLM  (y = 7.5)
    # ══════════════════════════════════════════════════════════════════════
    Y_VLM = 7.5

    # Vision Tokens → VLM
    draw_arrow(ax, 9.5,  Y_VIS - 0.38, 11.0, Y_VLM + 0.38,
               label="Vision\nTokens", label_side="left")

    # Text Tokens → VLM merge point
    draw_arrow(ax, 5.1,  Y_TXT + 0.38, 11.0, Y_VLM - 0.38,
               label="Text\nTokens", label_side="right")

    # VLM block
    draw_block(ax, 12.0, Y_VLM, 2.2, 1.1,
               "VLM",
               "Qwen3-VL-8B\nhidden=4096, 36 layers\nGQA  ·  ~8B params",
               color=COL["vlm"], edge_color=_darken(COL["vlm"]),
               fontsize=11)

    draw_arrow(ax, 11.0, Y_VLM, 10.9, Y_VLM)   # leading arrow into VLM
    draw_arrow(ax, 10.9, Y_VLM, 10.9, Y_VLM)

    # Expert Tokens
    draw_block(ax, 15.0, Y_VLM, 1.8, 0.80,
               "Expert Tokens", "seq × 4096",
               color=COL["input"], edge_color=_darken(COL["input"]))

    draw_arrow(ax, 13.1, Y_VLM, 14.1, Y_VLM,
               label="Reasoning\noutput")

    # ══════════════════════════════════════════════════════════════════════
    # ROW D — Diffusion pipeline  (y = 3.8)
    # ══════════════════════════════════════════════════════════════════════
    Y_DIFF = 3.8

    # Noise input
    draw_block(ax, 1.2, Y_DIFF, 1.4, 0.75,
               "Gaussian Noise", r"$\epsilon \sim \mathcal{N}(0,I)$" "\n64×2",
               color=COL["input"], edge_color=_darken(COL["input"]),
               fontsize=9.5)

    # Expert Decoder (conditioning)
    draw_block(ax, 4.4, Y_DIFF, 2.2, 1.0,
               "Expert Decoder",
               "hidden=2048, 16 heads\nCross-Attention\n(conditions on Expert Tokens)",
               color=COL["expert"], edge_color=_darken(COL["expert"]),
               fontsize=10)

    # Arrow: Expert Tokens → Expert Decoder
    draw_arrow(ax, 15.0, Y_VLM - 0.40, 5.5, Y_DIFF + 0.50,
               label="Expert\nTokens")

    # Diffusion loop
    draw_block(ax, 8.2, Y_DIFF, 2.8, 1.1,
               "Diffusion Loop",
               "Flow Matching  ·  10 Euler steps\nU-Net / DiT denoiser\nDecoder conditioning injected",
               color=COL["diffusion"], edge_color=_darken(COL["diffusion"]),
               fontsize=10)

    draw_arrow(ax, 1.9, Y_DIFF, 3.3, Y_DIFF,
               label="noisy\nactions")
    draw_arrow(ax, 5.5, Y_DIFF, 6.8, Y_DIFF,
               label="conditioning\nvectors")
    draw_arrow(ax, 9.6, Y_DIFF, 11.1, Y_DIFF,
               label="denoised\n64×2")

    # Denoised Actions
    draw_block(ax, 12.0, Y_DIFF, 1.8, 0.80,
               "Denoised Actions", "shape (64, 2)\n[accel, curvature]",
               color=COL["action"], edge_color=_darken(COL["action"]))

    # ══════════════════════════════════════════════════════════════════════
    # ROW E — Unicycle model + Trajectory  (y = 1.6)
    # ══════════════════════════════════════════════════════════════════════
    Y_UNI = 1.6

    draw_arrow(ax, 12.0, Y_DIFF - 0.40, 12.0, Y_UNI + 0.55)

    draw_block(ax, 12.0, Y_UNI, 2.2, 1.0,
               "Unicycle Model",
               r"$x_{k+1}=x_k+v_k\cos\psi_k\,\Delta t$" "\n"
               r"$y_{k+1}=y_k+v_k\sin\psi_k\,\Delta t$" "\n"
               r"$\psi_{k+1}=\psi_k+v_k\kappa_k\,\Delta t$",
               color=COL["action"], edge_color=_darken(COL["action"]),
               fontsize=9)

    draw_arrow(ax, 13.1, Y_UNI, 14.5, Y_UNI)

    draw_block(ax, 15.5, Y_UNI, 2.4, 1.0,
               "Trajectory",
               "shape (64, 3)\n[x,  y,  yaw]\ndt = 0.1 s  →  6.4 s horizon",
               color=COL["traj"], edge_color=_darken(COL["traj"]),
               fontsize=10)

    # ══════════════════════════════════════════════════════════════════════
    # Legend panel (bottom-left)
    # ══════════════════════════════════════════════════════════════════════
    legend_items = [
        ("Input / Intermediate Tensor", COL["input"]),
        ("Vision Encoder",              COL["vision"]),
        ("Text Processing",             COL["text"]),
        ("VLM  (Qwen3-VL-8B)",          COL["vlm"]),
        ("Expert Decoder",              COL["expert"]),
        ("Diffusion",                   COL["diffusion"]),
        ("Action / Unicycle",           COL["action"]),
        ("Output Trajectory",           COL["traj"]),
    ]

    lx, ly0 = 0.35, 4.8
    ax.text(lx, ly0 + 0.22, "Legend", fontsize=9, fontweight="bold",
            color="#37474F", va="center")
    for i, (lbl, col) in enumerate(legend_items):
        box = FancyBboxPatch(
            (lx - 0.18, ly0 - 0.48 - i * 0.52), 0.32, 0.32,
            boxstyle="round,pad=0.04",
            facecolor=col, edgecolor=_darken(col),
            linewidth=1.2, zorder=4,
        )
        ax.add_patch(box)
        ax.text(lx + 0.25, ly0 - 0.32 - i * 0.52, lbl,
                va="center", ha="left", fontsize=8.2, color="#212121")

    # ══════════════════════════════════════════════════════════════════════
    # Parameter count summary box (bottom-right)
    # ══════════════════════════════════════════════════════════════════════
    summary = (
        "Parameter Summary\n"
        "─────────────────────────\n"
        "Vision Encoder   ~427 M\n"
        "VLM (Qwen3-VL)  ~8 000 M\n"
        "Expert Decoder    ~200 M\n"
        "Diffusion         ~500 M\n"
        "─────────────────────────\n"
        "Total           ~9 127 M\n"
        "(≈ 10B, fp16 ≈ 18.3 GB)"
    )
    ax.text(
        16.8, 2.1, summary,
        ha="right", va="top",
        fontsize=8.2, color="white",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6",
                  fc="#37474F", ec="#263238", alpha=0.92),
        zorder=5,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Flow label annotations
    # ══════════════════════════════════════════════════════════════════════
    ax.text(9, 10.55,
            "[ Vision Pipeline ]",
            ha="center", va="center", fontsize=9,
            color=_darken(COL["vision"]), fontstyle="italic")
    ax.text(3.5, 6.75,
            "[ Text Pipeline ]",
            ha="center", va="center", fontsize=9,
            color=_darken(COL["text"]), fontstyle="italic")
    ax.text(12.0, 8.35,
            "[ VLM Backbone ]",
            ha="center", va="center", fontsize=9,
            color=_darken(COL["vlm"]), fontstyle="italic")
    ax.text(6.5, 4.82,
            "[ Diffusion  +  Expert Conditioning ]",
            ha="center", va="center", fontsize=9,
            color=_darken(COL["diffusion"]), fontstyle="italic")
    ax.text(13.8, 0.72,
            "[ Action Decoding  →  Trajectory ]",
            ha="center", va="center", fontsize=9,
            color=_darken(COL["traj"]), fontstyle="italic")

    return fig


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib

    OUT = pathlib.Path(
        "/home/seungwoo/workspace/analysis/figures/architecture_overview.png"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    fig.savefig(str(OUT), dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {OUT}")
