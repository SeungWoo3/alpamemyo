"""
visualize_architecture.py

Comprehensive architecture block diagram for Alpamayo-R1-10B.

Layout: vertical flow, top to bottom
  1. Image Input
  2. Vision Encoder  (SigLIP-400M-based)
       PatchEmbed: Conv3d(3 to 1152, k=2x16x16)
       27x VisionBlock: LN -> Attn(16h, 1152) -> MLP(1152->4304->1152)
       PatchMerger: 2x2 merge + MLP(4608->4096)
  3. VLM  (Qwen3-VL-8B)
       36 Transformer layers, hidden=4096, 32 heads / 8 KV (GQA)
       intermediate=12288
       Input: [vision tokens + text/history tokens]
       Output: KV cache for Expert conditioning
  4. Expert Decoder  (~2B)
       16 Transformer layers, hidden=2048, 16 heads, intermediate=8256
       Uses VLM KV cache as context
  5. Diffusion Head
       action_in_proj: FourierEncode + MLP -> (B, 64, 2048)
       10 Euler steps through Expert Decoder
       action_out_proj: Linear(2048->2) -> velocity field
  6. Action Space / Trajectory Output
       (accel, curvature) -> Unicycle kinematics -> (x, y, yaw)
       64 waypoints, dt=0.1s

Saved to: /home/seungwoo/workspace/analysis/figures/architecture_overview.png
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
def _darken(hex_color: str, amount: float = 0.30) -> str:
    """Return a darkened version of a hex colour string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = max(0, int(r * (1 - amount)))
    g = max(0, int(g * (1 - amount)))
    b = max(0, int(b * (1 - amount)))
    return f"#{r:02x}{g:02x}{b:02x}"


C = {
    "bg"       : "#F5F7FA",
    "input"    : "#B3D9FF",   # light blue   — I/O tensors
    "vision"   : "#D7BDE2",   # lavender     — Vision Encoder
    "vlm"      : "#FAD7A0",   # peach        — VLM
    "expert"   : "#A9DFBF",   # light green  — Expert Decoder
    "diffusion": "#FADBD8",   # light pink   — Diffusion Head
    "action"   : "#AED6F1",   # sky blue     — Action space
    "traj"     : "#F9E79F",   # yellow       — Trajectory output
    "arrow"    : "#2C3E50",   # dark slate
    "subblock" : "#FDFEFE",   # near-white sub-block fill
    "math"     : "#EBF5FB",
}


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _block(ax, cx, cy, w, h,
           title, detail="",
           color=C["input"], lw=1.8,
           title_fs=10, detail_fs=8.2,
           zorder=3, radius=0.18):
    """Rounded rectangle with a title and optional detail text."""
    ec = _darken(color)
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    if detail:
        ax.text(cx, cy + h * 0.13, title,
                ha="center", va="center",
                fontsize=title_fs, fontweight="bold", zorder=zorder + 1)
        ax.text(cx, cy - h * 0.20, detail,
                ha="center", va="center",
                fontsize=detail_fs, color="#424242",
                linespacing=1.35, zorder=zorder + 1)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center",
                fontsize=title_fs, fontweight="bold", zorder=zorder + 1)


def _varrow(ax, x, y0, y1, label="", label_x_offset=0.18,
            color=C["arrow"], lw=1.6, mutation_scale=14, zorder=2):
    """Vertical downward arrow with optional side label."""
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=mutation_scale,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=zorder)
    if label:
        my = (y0 + y1) / 2
        ax.text(x + label_x_offset, my, label,
                ha="left", va="center",
                fontsize=7.8, color="#37474F",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
                zorder=zorder + 1)


def _harrow(ax, x0, x1, y, label="", label_y_offset=0.12,
            color=C["arrow"], lw=1.6, mutation_scale=14, zorder=2):
    """Horizontal arrow with optional label above."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=mutation_scale,
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=zorder)
    if label:
        mx = (x0 + x1) / 2
        ax.text(mx, y + label_y_offset, label,
                ha="center", va="bottom",
                fontsize=7.8, color="#37474F",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
                zorder=zorder + 1)


def _param_badge(ax, cx, cy, text, zorder=6):
    """Small dark badge for parameter count annotation."""
    ax.text(cx, cy, text,
            ha="center", va="center",
            fontsize=7.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="#37474F", ec="#263238",
                      linewidth=1.0),
            zorder=zorder)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------
def build_figure() -> plt.Figure:
    """
    Vertical flow diagram.

    Canvas: x in [0, 16], y in [0, 20]  (figure 16 x 20 inches)
    Flow:  y decreases downward  (top = 19.5, bottom = 0.5)

    Vertical positions (y centre of each major block):
      Image Input        : 19.0
      Vision Encoder     : 16.5  (group: 14.5 to 18.5)
      VLM                : 12.5  (group: 10.5 to 14.0)
      Expert Decoder     :  8.0  (group:  6.5 to  9.5)
      Diffusion Head     :  4.5  (group:  3.0 to  6.0)
      Action / Traj      :  1.5
    """
    fig, ax = plt.subplots(figsize=(16, 20))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.axis("off")

    CX = 8.0   # horizontal centre of main column

    # ============================================================
    # Header
    # ============================================================
    ax.text(CX, 19.55,
            "Alpamayo-R1-10B  —  Architecture Overview",
            ha="center", va="center",
            fontsize=16, fontweight="bold", color="#1A237E")
    ax.text(CX, 19.15,
            "Vision Encoder (SigLIP-400M)  |  VLM Qwen3-VL-8B  |  "
            "Expert Decoder (~2B)  |  Flow Matching Diffusion",
            ha="center", va="center",
            fontsize=9.5, color="#546E7A")

    # ============================================================
    # 0. Image Input
    # ============================================================
    Y_IMG = 18.3
    _block(ax, CX, Y_IMG, 3.8, 0.65,
           "Image Input",
           "H x W x 3  (camera frames)",
           color=C["input"], title_fs=10, detail_fs=8.5)

    # ============================================================
    # 1. Vision Encoder group  (y: 14.5 to 17.7)
    # ============================================================
    Y_VE_TOP  = 17.7
    Y_VE_BOT  = 14.5

    # Group background panel
    group_bg = FancyBboxPatch(
        (2.5, Y_VE_BOT - 0.10), 11.0, Y_VE_TOP - Y_VE_BOT + 0.20,
        boxstyle="round,pad=0.06",
        facecolor="#F3EBF9", edgecolor=_darken(C["vision"], 0.20),
        linewidth=1.4, zorder=1, linestyle="--",
    )
    ax.add_patch(group_bg)
    ax.text(2.65, Y_VE_TOP - 0.05,
            "Vision Encoder  (SigLIP-400M-based,  ~427M params)",
            ha="left", va="top",
            fontsize=9, color=_darken(C["vision"]), fontstyle="italic")

    # PatchEmbed
    Y_PE = 17.05
    _block(ax, CX, Y_PE, 9.0, 0.80,
           "PatchEmbed",
           "Conv3d(3 -> 1152,  kernel 2x16x16,  stride 2x16x16)\n"
           "Input frames -> spatial patch tokens  (B, N_patches, 1152)",
           color=C["vision"], title_fs=10, detail_fs=8.0)
    _param_badge(ax, CX + 4.8, Y_PE, "~10M")

    # 27x VisionBlock
    Y_VB = 15.85
    _block(ax, CX, Y_VB, 9.0, 1.50,
           "27x  VisionBlock",
           "LayerNorm  ->  Multi-Head Attention (16 heads, dim=1152)\n"
           "->  LayerNorm  ->  MLP (1152 -> 4304 -> 1152,  SiLU)\n"
           "RoPE positional encoding  |  hidden=1152",
           color=C["vision"], title_fs=10.5, detail_fs=8.2)
    _param_badge(ax, CX + 4.8, Y_VB, "~400M")

    # PatchMerger
    Y_PM = 14.85
    _block(ax, CX, Y_PM, 9.0, 0.75,
           "PatchMerger",
           "2x2 spatial merge  ->  MLP (4608 -> 4096)  |  output: (B, N/4, 4096)",
           color=C["vision"], title_fs=10, detail_fs=8.0)
    _param_badge(ax, CX + 4.8, Y_PM, "~37M")

    # Internal arrows (within Vision Encoder)
    _varrow(ax, CX, Y_PE + 0.40, Y_PE + 0.70,
            label="(B, N, 1152)", label_x_offset=0.20, lw=1.4)
    _varrow(ax, CX, Y_VB + 0.75, Y_PE - 0.40,
            label="(B, N, 1152)", label_x_offset=0.20, lw=1.4)
    _varrow(ax, CX, Y_PM + 0.38, Y_VB - 0.75,
            label="(B, N, 1152)", label_x_offset=0.20, lw=1.4)

    # Arrow: Image Input -> PatchEmbed
    _varrow(ax, CX, Y_IMG - 0.33, Y_PE + 0.40,
            label="HxWx3", label_x_offset=0.20)

    # ============================================================
    # 2. Text / History tokens (side input entering VLM)
    # ============================================================
    Y_TEXT = 13.5
    _block(ax, 2.5, Y_TEXT, 3.2, 0.65,
           "Text / History Tokens",
           "Language instruction\n+ ego-motion history",
           color=C["input"], title_fs=9, detail_fs=7.8)

    # ============================================================
    # 3. VLM  (y: 10.5 to 14.0)
    # ============================================================
    Y_VLM_TOP = 14.0
    Y_VLM_BOT = 10.5

    group_bg_vlm = FancyBboxPatch(
        (2.5, Y_VLM_BOT - 0.10), 11.0, Y_VLM_TOP - Y_VLM_BOT + 0.20,
        boxstyle="round,pad=0.06",
        facecolor="#FFF8EE", edgecolor=_darken(C["vlm"], 0.20),
        linewidth=1.4, zorder=1, linestyle="--",
    )
    ax.add_patch(group_bg_vlm)
    ax.text(2.65, Y_VLM_TOP - 0.05,
            "VLM  (Qwen3-VL-8B,  ~8B params)",
            ha="left", va="top",
            fontsize=9, color=_darken(C["vlm"]), fontstyle="italic")

    Y_VLM_INPUT = 13.50
    _block(ax, CX, Y_VLM_INPUT, 9.0, 0.70,
           "Input Sequence",
           "[Vision tokens  |  Text tokens  |  Ego-history tokens]  ->  (B, L, 4096)",
           color=C["vlm"], title_fs=9.5, detail_fs=8.0)

    Y_VLM_LAYERS = 12.35
    _block(ax, CX, Y_VLM_LAYERS, 9.0, 1.60,
           "36x  Transformer Layer",
           "RMSNorm  ->  GQA Self-Attention (32 Q-heads / 8 KV-heads)  ->  RMSNorm\n"
           "->  SwiGLU FFN (hidden=4096, intermediate=12288)\n"
           "RoPE positional encoding  |  Sliding-window + full attention",
           color=C["vlm"], title_fs=10.5, detail_fs=8.2)
    _param_badge(ax, CX + 4.8, Y_VLM_LAYERS, "~8B")

    Y_VLM_OUT = 10.85
    _block(ax, CX, Y_VLM_OUT, 9.0, 0.70,
           "VLM Output  ->  KV Cache",
           "KV cache: (B, 36, L, 4096/8-heads)  |  used as context for Expert Decoder",
           color=C["vlm"], title_fs=9.5, detail_fs=8.0)

    # Arrows within VLM
    _varrow(ax, CX, Y_VLM_LAYERS + 0.80, Y_VLM_INPUT - 0.35,
            label="(B, L, 4096)", label_x_offset=0.20, lw=1.4)
    _varrow(ax, CX, Y_VLM_OUT + 0.35, Y_VLM_LAYERS - 0.80,
            label="(B, L, 4096)", label_x_offset=0.20, lw=1.4)

    # Arrow: PatchMerger -> VLM input
    _varrow(ax, CX, Y_PM - 0.38, Y_VLM_INPUT + 0.35,
            label="(B, N/4, 4096)", label_x_offset=0.20)

    # Arrow: Text tokens -> VLM input (diagonal from left)
    ax.annotate("", xy=(CX - 3.5, Y_VLM_INPUT),
                xytext=(2.5 + 1.6, Y_TEXT),
                arrowprops=dict(
                    arrowstyle="-|>", color=C["arrow"], lw=1.5,
                    mutation_scale=13,
                    connectionstyle="arc3,rad=-0.15",
                ),
                zorder=2)
    ax.text(4.0, (Y_VLM_INPUT + Y_TEXT) / 2 + 0.15,
            "text +\nhistory",
            ha="right", va="center", fontsize=7.5, color="#37474F",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    # ============================================================
    # 4. Expert Decoder  (y: 6.5 to 9.5)
    # ============================================================
    Y_EX_TOP = 9.5
    Y_EX_BOT = 6.5
    Y_EX_MID = (Y_EX_TOP + Y_EX_BOT) / 2

    group_bg_ex = FancyBboxPatch(
        (2.5, Y_EX_BOT - 0.10), 11.0, Y_EX_TOP - Y_EX_BOT + 0.20,
        boxstyle="round,pad=0.06",
        facecolor="#EAFAF1", edgecolor=_darken(C["expert"], 0.25),
        linewidth=1.4, zorder=1, linestyle="--",
    )
    ax.add_patch(group_bg_ex)
    ax.text(2.65, Y_EX_TOP - 0.05,
            "Expert Decoder  (~2B params)",
            ha="left", va="top",
            fontsize=9, color=_darken(C["expert"]), fontstyle="italic")

    Y_EX_LAYERS = 8.2
    _block(ax, CX, Y_EX_LAYERS, 9.0, 2.0,
           "16x  Transformer Layer",
           "RMSNorm  ->  Self-Attention (16 heads, hidden=2048)\n"
           "->  Cross-Attention conditioned on VLM KV cache\n"
           "->  SwiGLU FFN (hidden=2048, intermediate=8256)\n"
           "|  non-causal (expert_non_causal_attention=True)",
           color=C["expert"], title_fs=10.5, detail_fs=8.2)
    _param_badge(ax, CX + 4.8, Y_EX_LAYERS, "~2B")

    Y_EX_OUT = 6.85
    _block(ax, CX, Y_EX_OUT, 9.0, 0.60,
           "Expert Output",
           "last_hidden_state  ->  (B, 64, 2048)",
           color=C["expert"], title_fs=9.5, detail_fs=8.0)

    # Arrows within Expert Decoder
    _varrow(ax, CX, Y_EX_OUT + 0.30, Y_EX_LAYERS - 1.0,
            label="(B, 64, 2048)", label_x_offset=0.20, lw=1.4)
    _varrow(ax, CX, Y_EX_LAYERS + 1.0, Y_EX_OUT + 0.60,
            label="(B, 64, 2048)", label_x_offset=0.20, lw=1.4)

    # KV cache feed from VLM to Expert Decoder (dashed side arrow)
    kv_x = 13.8
    ax.annotate("", xy=(kv_x, Y_EX_LAYERS),
                xytext=(kv_x, Y_VLM_OUT - 0.35),
                arrowprops=dict(
                    arrowstyle="-|>", color=_darken(C["vlm"]),
                    lw=1.8, mutation_scale=14,
                    linestyle="dashed",
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=2)
    ax.text(kv_x + 0.15, (Y_VLM_OUT + Y_EX_LAYERS) / 2,
            "KV cache\n(VLM -> Expert)",
            ha="left", va="center", fontsize=8, color=_darken(C["vlm"]),
            fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    # Arrow: VLM output -> Expert Decoder (main spine)
    _varrow(ax, CX, Y_VLM_OUT - 0.35, Y_EX_LAYERS + 1.0,
            label="(B, 64, 2048)\naction embeds",
            label_x_offset=0.20)

    # ============================================================
    # 5. Diffusion Head  (y: 3.0 to 6.0)
    # ============================================================
    Y_DH_TOP = 6.0
    Y_DH_BOT = 3.0

    group_bg_dh = FancyBboxPatch(
        (2.5, Y_DH_BOT - 0.10), 11.0, Y_DH_TOP - Y_DH_BOT + 0.20,
        boxstyle="round,pad=0.06",
        facecolor="#FEF5F5", edgecolor=_darken(C["diffusion"], 0.25),
        linewidth=1.4, zorder=1, linestyle="--",
    )
    ax.add_patch(group_bg_dh)
    ax.text(2.65, Y_DH_TOP - 0.05,
            "Diffusion Head  (Flow Matching, 10 Euler steps)",
            ha="left", va="top",
            fontsize=9, color=_darken(C["diffusion"]), fontstyle="italic")

    # Noise input (side entry)
    Y_NOISE = 5.35
    _block(ax, 2.1, Y_NOISE, 2.6, 0.65,
           "Gaussian Noise",
           "x0 ~ N(0, I)\n(B, 64, 2)",
           color=C["input"], title_fs=9, detail_fs=8.0)

    # action_in_proj
    Y_AINP = 5.35
    _block(ax, CX, Y_AINP, 8.5, 0.72,
           "action_in_proj",
           "FourierEncode(x, t)  ->  MLP (hidden=1024, 4 layers)  ->  LayerNorm\n"
           "Output: (B, 64, 2048)  [action + timestep embeddings]",
           color=C["diffusion"], title_fs=10, detail_fs=8.0)

    # 10 Euler steps loop
    Y_EULER = 4.35
    _block(ax, CX, Y_EULER, 8.5, 1.00,
           "10x Euler Step:  x = x + dt * v",
           "dt = 0.1,   t = {0.0, 0.1, ..., 0.9}\n"
           "Each step: action_in_proj -> Expert Decoder (16 layers) -> action_out_proj",
           color=C["diffusion"], title_fs=10, detail_fs=8.3)

    # action_out_proj
    Y_AOUTP = 3.40
    _block(ax, CX, Y_AOUTP, 8.5, 0.68,
           "action_out_proj",
           "Linear (2048 -> 2)  ->  velocity field  v  |  Output: (B, 64, 2)",
           color=C["diffusion"], title_fs=10, detail_fs=8.0)

    # Arrows: noise -> action_in_proj
    _harrow(ax, 2.1 + 1.3, CX - 4.25, Y_AINP,
            label="(B,64,2)", label_y_offset=0.10, lw=1.4)

    # Arrows within diffusion head
    _varrow(ax, CX, Y_AINP - 0.36, Y_EULER + 0.50,
            label="(B, 64, 2048)", label_x_offset=0.20, lw=1.4)
    _varrow(ax, CX, Y_EULER - 0.50, Y_AOUTP + 0.34,
            label="(B, 64, 2048)", label_x_offset=0.20, lw=1.4)

    # Arrow: Expert Decoder output -> Diffusion Head
    _varrow(ax, CX, Y_EX_OUT - 0.30, Y_AINP + 0.36,
            label="(B, 64, 2048)", label_x_offset=0.20)

    # Feedback loop label (Expert Decoder used inside loop)
    ax.annotate("", xy=(13.7, Y_EULER),
                xytext=(13.7, Y_EX_MID),
                arrowprops=dict(
                    arrowstyle="-|>", color=_darken(C["expert"]),
                    lw=1.6, mutation_scale=12,
                    linestyle="dashed",
                    connectionstyle="arc3,rad=0.0",
                ),
                zorder=2)
    ax.text(13.85, (Y_EULER + Y_EX_MID) / 2,
            "Expert Decoder\ncalled per step",
            ha="left", va="center", fontsize=7.5,
            color=_darken(C["expert"]), fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))

    # ============================================================
    # 6. Action Space & Trajectory Output
    # ============================================================
    Y_ACT = 2.45
    _block(ax, CX, Y_ACT, 9.0, 0.72,
           "Action Space  ->  Unicycle Kinematics",
           "(accel, curvature) x 64  ->  v, theta, x, y\n"
           "x[k+1] = x[k] + v[k]*cos(theta[k])*dt,   dt=0.1 s",
           color=C["action"], title_fs=10, detail_fs=8.2)

    Y_TRAJ = 1.55
    _block(ax, CX, Y_TRAJ, 9.0, 0.68,
           "Trajectory Output",
           "shape (B, 64, 3)  |  [x (m),  y (m),  yaw (rad)]\n"
           "64 waypoints at dt=0.1 s  ->  6.4 s prediction horizon",
           color=C["traj"], title_fs=10, detail_fs=8.2)

    # Arrows
    _varrow(ax, CX, Y_AOUTP - 0.34, Y_ACT + 0.36,
            label="(B, 64, 2)\n[accel, curv]",
            label_x_offset=0.20)
    _varrow(ax, CX, Y_ACT - 0.36, Y_TRAJ + 0.34,
            label="(B, 64, 3)", label_x_offset=0.20)

    # ============================================================
    # Parameter summary box  (right side)
    # ============================================================
    summary = (
        "Parameter Summary\n"
        "-----------------------------\n"
        "Vision Encoder        ~427 M\n"
        "  PatchEmbed           ~10 M\n"
        "  27x VisionBlock     ~400 M\n"
        "  PatchMerger          ~37 M\n"
        "VLM  (Qwen3-VL-8B)  ~8,000 M\n"
        "Expert Decoder        ~200 M\n"
        "Diffusion Head         ~50 M\n"
        "  action_in_proj       ~30 M\n"
        "  action_out_proj       ~4 M\n"
        "-----------------------------\n"
        "Total              ~8,677 M\n"
        "(approx 10B,  FP16 ~17.4 GB)"
    )
    ax.text(
        15.85, 11.8, summary,
        ha="right", va="top",
        fontsize=7.8, color="white", fontweight="normal",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.60",
                  fc="#2C3E50", ec="#1A252F",
                  linewidth=1.2, alpha=0.93),
        zorder=6,
    )

    # ============================================================
    # Colour legend  (bottom left)
    # ============================================================
    legend_items = [
        ("Input / Intermediate Tensor", C["input"]),
        ("Vision Encoder  (SigLIP)",    C["vision"]),
        ("VLM  (Qwen3-VL-8B)",          C["vlm"]),
        ("Expert Decoder  (~2B)",        C["expert"]),
        ("Diffusion Head",               C["diffusion"]),
        ("Action Space  /  Kinematics", C["action"]),
        ("Trajectory Output",            C["traj"]),
    ]

    lx, ly0 = 0.30, 4.80
    ax.text(lx + 0.08, ly0 + 0.35, "Legend",
            fontsize=9, fontweight="bold", color="#37474F", va="center")
    dy = 0.55
    for i, (lbl, col) in enumerate(legend_items):
        sw = FancyBboxPatch(
            (lx, ly0 - 0.18 - i * dy), 0.38, 0.34,
            boxstyle="round,pad=0.04",
            facecolor=col, edgecolor=_darken(col),
            linewidth=1.1, zorder=4,
        )
        ax.add_patch(sw)
        ax.text(lx + 0.52, ly0 - 0.01 - i * dy, lbl,
                va="center", ha="left", fontsize=8.2, color="#212121")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_path = "/home/seungwoo/workspace/analysis/figures/architecture_overview.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = build_figure()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
