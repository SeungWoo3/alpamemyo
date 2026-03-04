"""
Vision Encoder Architecture Visualization for Alpamayo-R1-10B.

Draws a detailed block diagram of:
  PatchEmbed (Conv3d) → 27× VisionBlock → PatchMerger

Saved to: analysis/figures/vision_encoder_detail.png at 300 DPI.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.left"] = False
matplotlib.rcParams["axes.spines.bottom"] = False


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG        = "#F7F9FC"   # figure background
C_EMBED     = "#E67E22"   # orange  — PatchEmbed
C_ATTN      = "#2980B9"   # blue    — MultiHead Attention
C_MLP       = "#27AE60"   # green   — MLP
C_NORM      = "#95A5A6"   # grey    — LayerNorm
C_RESIDUAL  = "#BDC3C7"   # light grey — residual connections
C_MERGER    = "#8E44AD"   # purple  — PatchMerger
C_BLOCK_BG  = "#EAF4FF"   # very light blue — VisionBlock outline
C_ARROW     = "#2C3E50"   # dark    — arrows
C_DIM       = "#7F8C8D"   # dim label text
C_WHITE     = "#FFFFFF"
C_TITLE     = "#1A252F"


# ---------------------------------------------------------------------------
# Helper: draw a rounded rectangle with centred text
# ---------------------------------------------------------------------------
def draw_box(ax, x, y, w, h, label, sub_label=None,
             facecolor=C_WHITE, edgecolor=C_ARROW,
             fontsize=9, sub_fontsize=7.5, lw=1.5, radius=0.015,
             text_color="black", bold=False):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if sub_label:
        ax.text(x, y + h * 0.12, label,
                ha="center", va="center", fontsize=fontsize,
                color=text_color, fontweight=weight, zorder=4)
        ax.text(x, y - h * 0.22, sub_label,
                ha="center", va="center", fontsize=sub_fontsize,
                color=C_DIM, zorder=4, style="italic")
    else:
        ax.text(x, y, label,
                ha="center", va="center", fontsize=fontsize,
                color=text_color, fontweight=weight, zorder=4)
    return box


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.4, style="->"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color,
            lw=lw, connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )


def dim_label(ax, x, y, text, fontsize=7, color=C_DIM, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, style="italic", zorder=6)


# ---------------------------------------------------------------------------
# Layout constants  (all in "axes fraction" units 0..1 mapped to data coords)
# We work in data coords directly; figure is 16 × 10 inches.
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 16, 10   # inches

# We use data coordinates where the axes span [0, 16] × [0, 10].
# The axes fills the whole figure (no margins — we set them manually).
AX_XLIM = (0, 16)
AX_YLIM = (0, 10)


def make_figure():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(*AX_XLIM)
    ax.set_ylim(*AX_YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
    return fig, ax


# ---------------------------------------------------------------------------
# Main drawing
# ---------------------------------------------------------------------------
def draw_vision_encoder():
    fig, ax = make_figure()

    # --- Title ---
    ax.text(8, 9.6, "Alpamayo-R1-10B  |  Vision Encoder Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_TITLE, zorder=6)
    ax.text(8, 9.25, "PatchEmbed → 27× VisionBlock → PatchMerger",
            ha="center", va="center", fontsize=9, color=C_DIM, zorder=6)

    # -----------------------------------------------------------------------
    # Column layout (x centres):
    #   Input  PatchEmbed  VisionBlock-repeat  PatchMerger  Output
    # -----------------------------------------------------------------------
    X_INPUT   = 1.0
    X_EMBED   = 2.8
    X_BLOCK   = 8.0    # centre of the VisionBlock region
    X_MERGER  = 13.3
    X_OUTPUT  = 15.1

    # We'll draw everything at y centre Y_MID, with the block spanning
    # most of the height.
    Y_MID  = 4.8
    Y_TOP  = 8.6
    Y_BOT  = 1.0

    # --- INPUT box ---
    draw_box(ax, X_INPUT, Y_MID, 1.3, 0.7,
             "Input Video",
             sub_label="(3, T, H, W)",
             facecolor="#FDFEFE", edgecolor=C_ARROW, fontsize=8.5)

    # arrow input → embed
    arrow(ax, X_INPUT + 0.65, Y_MID, X_EMBED - 0.9, Y_MID)

    # --- PATCHEMBEDDING box ---
    draw_box(ax, X_EMBED, Y_MID, 1.6, 1.6,
             "PatchEmbed",
             sub_label="Conv3d(3,1152)\nk=(2,16,16)\ns=(2,16,16)",
             facecolor=C_EMBED, edgecolor="#BA6011",
             fontsize=9, sub_fontsize=7, text_color=C_WHITE, bold=True,
             lw=2)
    dim_label(ax, X_EMBED, Y_MID - 1.1,
              "3 × H × W → N patches × 1152",
              fontsize=7)

    # arrow embed → block region
    arrow(ax, X_EMBED + 0.8, Y_MID, 4.15, Y_MID)
    dim_label(ax, 3.8, Y_MID + 0.3, "(N, 1152)", fontsize=7)

    # -----------------------------------------------------------------------
    # VisionBlock region — outer dashed rectangle
    # -----------------------------------------------------------------------
    BLOCK_X0 = 4.2
    BLOCK_X1 = 11.8
    BLOCK_Y0 = Y_BOT - 0.1
    BLOCK_Y1 = Y_TOP + 0.35

    repeat_rect = FancyBboxPatch(
        (BLOCK_X0, BLOCK_Y0),
        BLOCK_X1 - BLOCK_X0,
        BLOCK_Y1 - BLOCK_Y0,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=C_BLOCK_BG, edgecolor=C_ATTN,
        linewidth=2, linestyle="--", zorder=1,
    )
    ax.add_patch(repeat_rect)
    ax.text((BLOCK_X0 + BLOCK_X1) / 2, BLOCK_Y1 - 0.17,
            "× 27   VisionBlock",
            ha="center", va="center", fontsize=9.5,
            color=C_ATTN, fontweight="bold", zorder=4)

    # -----------------------------------------------------------------------
    # Inside VisionBlock: vertical stack of sub-components
    # We arrange them top-to-bottom, with residual connections shown on the
    # right side.
    # -----------------------------------------------------------------------
    # Sub-component x centre inside block
    XC = (BLOCK_X0 + BLOCK_X1) / 2   # 8.0
    BOX_W = 3.0    # width of inner boxes
    BOX_H = 0.72   # height of inner boxes

    # y positions from top to bottom
    y_ln1   = 8.0
    y_attn  = 6.8
    y_add1  = 5.85   # "Add & Norm" (residual 1)
    y_ln2   = 4.85
    y_mlp   = 3.65
    y_add2  = 2.65   # "Add & Norm" (residual 2)

    # LayerNorm 1
    draw_box(ax, XC, y_ln1, BOX_W, BOX_H,
             "LayerNorm",
             sub_label="dim=1152",
             facecolor=C_NORM, edgecolor="#707B7C",
             text_color=C_WHITE, fontsize=9)

    # MultiHead Attention
    draw_box(ax, XC, y_attn, BOX_W, BOX_H + 0.18,
             "Multi-Head Self-Attention",
             sub_label="heads=16, head_dim=72  (1152→1152)",
             facecolor=C_ATTN, edgecolor="#1A5276",
             text_color=C_WHITE, fontsize=9, bold=True, lw=2)

    # Residual Add 1
    draw_box(ax, XC, y_add1, BOX_W, BOX_H - 0.12,
             "Residual Add  ( x + Attn(LN(x)) )",
             facecolor=C_RESIDUAL, edgecolor="#95A5A6",
             fontsize=8)

    # LayerNorm 2
    draw_box(ax, XC, y_ln2, BOX_W, BOX_H,
             "LayerNorm",
             sub_label="dim=1152",
             facecolor=C_NORM, edgecolor="#707B7C",
             text_color=C_WHITE, fontsize=9)

    # MLP
    draw_box(ax, XC, y_mlp, BOX_W, BOX_H + 0.18,
             "MLP  (SiLU activation)",
             sub_label="1152 → 4304 → 1152",
             facecolor=C_MLP, edgecolor="#1A6B3C",
             text_color=C_WHITE, fontsize=9, bold=True, lw=2)

    # Residual Add 2
    draw_box(ax, XC, y_add2, BOX_W, BOX_H - 0.12,
             "Residual Add  ( x + MLP(LN(x)) )",
             facecolor=C_RESIDUAL, edgecolor="#95A5A6",
             fontsize=8)

    # Arrows inside block (vertical chain)
    connections = [
        (y_ln1 - BOX_H / 2,   y_attn + (BOX_H + 0.18) / 2),
        (y_attn - (BOX_H + 0.18) / 2, y_add1 + (BOX_H - 0.12) / 2),
        (y_add1 - (BOX_H - 0.12) / 2, y_ln2 + BOX_H / 2),
        (y_ln2 - BOX_H / 2,   y_mlp + (BOX_H + 0.18) / 2),
        (y_mlp - (BOX_H + 0.18) / 2,  y_add2 + (BOX_H - 0.12) / 2),
    ]
    for (y_from, y_to) in connections:
        arrow(ax, XC, y_from, XC, y_to)

    # Block input/output arrows (entering/leaving the block area)
    arrow(ax, XC, BLOCK_Y1 - 0.35 - 0.05, XC, y_ln1 + BOX_H / 2,
          style="-|>")

    # Residual bypass — first residual (skip from block input to Add1)
    X_SKIP_R = XC + BOX_W / 2 + 0.35
    # Drop from entry level to Add1 level
    y_entry = y_ln1 + BOX_H / 2 + 0.15   # just above LN1
    ax.plot([XC + BOX_W / 2, X_SKIP_R, X_SKIP_R,
             XC + BOX_W / 2],
            [y_entry, y_entry, y_add1, y_add1],
            color="#E74C3C", lw=1.5, zorder=5, linestyle="--")
    ax.annotate("", xy=(XC + BOX_W / 2, y_add1),
                xytext=(X_SKIP_R, y_add1),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.5),
                zorder=6)
    ax.text(X_SKIP_R + 0.08, (y_entry + y_add1) / 2,
            "skip\nconnection", ha="left", va="center",
            fontsize=6.5, color="#E74C3C", style="italic", zorder=6)

    # Residual bypass — second (skip from add1 to Add2)
    X_SKIP_L = XC - BOX_W / 2 - 0.35
    y_skip2_top = y_add1 - (BOX_H - 0.12) / 2 - 0.05
    ax.plot([XC - BOX_W / 2, X_SKIP_L, X_SKIP_L,
             XC - BOX_W / 2],
            [y_skip2_top, y_skip2_top, y_add2, y_add2],
            color="#E74C3C", lw=1.5, zorder=5, linestyle="--")
    ax.annotate("", xy=(XC - BOX_W / 2, y_add2),
                xytext=(X_SKIP_L, y_add2),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.5),
                zorder=6)

    # Arrow out of block
    arrow(ax, XC, y_add2 - (BOX_H - 0.12) / 2,
          XC, BLOCK_Y0 + 0.05, style="-|>")

    # dim annotation along right side of block
    dim_label(ax, BLOCK_X1 + 0.18, (BLOCK_Y0 + BLOCK_Y1) / 2,
              "shape preserved:\n(N, 1152)", fontsize=7, ha="left")

    # -----------------------------------------------------------------------
    # Arrow from block to merger
    # -----------------------------------------------------------------------
    arrow(ax, BLOCK_X1 + 0.05, Y_MID, X_MERGER - 0.85, Y_MID)
    dim_label(ax, 12.35, Y_MID + 0.3, "(N, 1152)", fontsize=7)

    # -----------------------------------------------------------------------
    # PatchMerger
    # -----------------------------------------------------------------------
    draw_box(ax, X_MERGER, Y_MID, 1.55, 2.1,
             "PatchMerger",
             sub_label="2×2 spatial merge\n4608 → 4096\n(MLP)",
             facecolor=C_MERGER, edgecolor="#6C3483",
             text_color=C_WHITE, fontsize=9, bold=True, lw=2)
    dim_label(ax, X_MERGER, Y_MID - 1.3,
              "N patches → N/4 tokens\n1152×4 = 4608 → 4096",
              fontsize=7)

    # Arrow merger → output
    arrow(ax, X_MERGER + 0.78, Y_MID, X_OUTPUT - 0.65, Y_MID)
    dim_label(ax, X_OUTPUT - 0.4, Y_MID + 0.3, "(N/4, 4096)", fontsize=7)

    # Output box
    draw_box(ax, X_OUTPUT, Y_MID, 1.3, 0.7,
             "Visual\nTokens",
             facecolor="#FDFEFE", edgecolor=C_ARROW, fontsize=8.5)

    # -----------------------------------------------------------------------
    # Dimension flow annotation at the bottom
    # -----------------------------------------------------------------------
    y_ann = 0.45
    ann_xs = [X_INPUT, X_EMBED, (BLOCK_X0 + BLOCK_X1) / 2, X_MERGER, X_OUTPUT]
    ann_labels = [
        "(3, T, H, W)",
        "→ (N, 1152)\nN = T/2 · H/16 · W/16",
        "→ (N, 1152)\n[unchanged × 27]",
        "→ (N/4, 4608)\n[merged]",
        "→ (N/4, 4096)\n[projected]",
    ]
    ax.axhline(y=y_ann + 0.15, xmin=0.03, xmax=0.97,
               color=C_DIM, lw=0.6, linestyle=":", zorder=1)
    for xi, lb in zip(ann_xs, ann_labels):
        ax.text(xi, y_ann, lb, ha="center", va="top",
                fontsize=6.8, color=C_DIM, style="italic",
                multialignment="center", zorder=5)

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    legend_items = [
        mpatches.Patch(facecolor=C_EMBED,  edgecolor="#BA6011", label="Patch Embedding (Conv3d)"),
        mpatches.Patch(facecolor=C_ATTN,   edgecolor="#1A5276", label="Multi-Head Attention"),
        mpatches.Patch(facecolor=C_MLP,    edgecolor="#1A6B3C", label="MLP (SiLU)"),
        mpatches.Patch(facecolor=C_NORM,   edgecolor="#707B7C", label="Layer Normalization"),
        mpatches.Patch(facecolor=C_RESIDUAL, edgecolor="#95A5A6", label="Residual Add"),
        mpatches.Patch(facecolor=C_MERGER, edgecolor="#6C3483", label="Patch Merger"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.005),
        fontsize=7.5,
        framealpha=0.9,
        edgecolor="#BDC3C7",
        ncol=2,
    )

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    out_path = "/home/seungwoo/workspace/analysis/figures/vision_encoder_detail.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = draw_vision_encoder()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
