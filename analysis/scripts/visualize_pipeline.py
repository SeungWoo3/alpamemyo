"""
Alpamayo-R1-10B Pipeline Visualization
Horizontal flowchart of the 7-stage inference pipeline.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9


# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
STAGE_W = 2.2          # box width
STAGE_H = 1.6          # box height
H_GAP = 0.7            # horizontal gap between boxes
Y_CENTER = 4.0         # vertical center of all boxes
FIG_W = 20.0
FIG_H = 8.0

STAGES = [
    {
        "title": "1. Patch\nEmbedding",
        "body": "Conv2D\n(patch=16)",
        "in_shape":  "Image\n(3, H, W)",
        "out_shape": "(N, 1152)",
        "color": "#4E79A7",
    },
    {
        "title": "2. Vision\nTransformer",
        "body": "27 Layers\nMHA h=16\nd_head=72",
        "in_shape":  "(N, 1152)",
        "out_shape": "(N, 1152)",
        "color": "#F28E2B",
    },
    {
        "title": "3. Patch\nMerger",
        "body": "2×2 Merge\nLinear Proj",
        "in_shape":  "(N, 1152)",
        "out_shape": "(N/4, 4096)",
        "color": "#E15759",
    },
    {
        "title": "4. VLM\nGeneration",
        "body": "Qwen3-VL-8B\n36 Layers\nGQA 32h/8kv",
        "in_shape":  "[vis+text]\n(S, 4096)",
        "out_shape": "Expert\n(S, 4096)",
        "color": "#76B7B2",
    },
    {
        "title": "5. Diffusion\nSampling",
        "body": "Flow Matching\n10 Euler steps",
        "in_shape":  "Noise\n(64, 2)",
        "out_shape": "Denoised\n(64, 2)",
        "color": "#59A14F",
    },
    {
        "title": "6. Action →\nTrajectory",
        "body": "Unicycle\nKinematics\ndt=0.1 s",
        "in_shape":  "(64, 2)\n[a, κ]",
        "out_shape": "(64, 3)\n[x,y,θ]",
        "color": "#EDC948",
    },
    {
        "title": "7. Final\nOutput",
        "body": "Trajectory\nTensor",
        "in_shape":  "(64, 3)",
        "out_shape": "64 waypoints\n× 3 dims",
        "color": "#B07AA1",
    },
]

N_STAGES = len(STAGES)
TOTAL_W = N_STAGES * STAGE_W + (N_STAGES - 1) * H_GAP
X_OFFSET = (FIG_W - TOTAL_W) / 2.0   # centering margin


def box_x(i: int) -> float:
    """Left edge x of the i-th stage box."""
    return X_OFFSET + i * (STAGE_W + H_GAP)


def draw_stage_box(ax, i, stage):
    x = box_x(i)
    y = Y_CENTER - STAGE_H / 2.0
    color = stage["color"]

    # Outer shadow
    shadow = FancyBboxPatch(
        (x + 0.05, y - 0.05), STAGE_W, STAGE_H,
        boxstyle="round,pad=0.08",
        linewidth=0,
        facecolor="#aaaaaa",
        alpha=0.35,
        zorder=1,
    )
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch(
        (x, y), STAGE_W, STAGE_H,
        boxstyle="round,pad=0.08",
        linewidth=1.5,
        edgecolor="white",
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(box)

    cx = x + STAGE_W / 2.0
    cy = Y_CENTER

    # Title (top half)
    ax.text(
        cx, cy + 0.35,
        stage["title"],
        ha="center", va="center",
        fontsize=9, fontweight="bold",
        color="white", zorder=3,
        linespacing=1.4,
    )

    # Divider line
    ax.plot(
        [x + 0.15, x + STAGE_W - 0.15],
        [cy, cy],
        color="white", linewidth=0.8, alpha=0.6, zorder=3,
    )

    # Body (bottom half)
    ax.text(
        cx, cy - 0.42,
        stage["body"],
        ha="center", va="center",
        fontsize=7.5,
        color="white", alpha=0.92, zorder=3,
        linespacing=1.35,
    )


def draw_arrow_with_labels(ax, i, stage_from, stage_to):
    """Draw arrow from box i to box i+1, with tensor shape labels."""
    x_start = box_x(i) + STAGE_W
    x_end = box_x(i + 1)
    mid_x = (x_start + x_end) / 2.0
    y = Y_CENTER

    # Arrow shaft
    ax.annotate(
        "",
        xy=(x_end, y),
        xytext=(x_start, y),
        arrowprops=dict(
            arrowstyle="->,head_width=0.25,head_length=0.18",
            color="#333333",
            lw=1.6,
        ),
        zorder=4,
    )

    # Tensor shape label above the arrow
    label = stage_from["out_shape"]
    ax.text(
        mid_x, y + 0.28,
        label,
        ha="center", va="bottom",
        fontsize=7,
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="#bbbbbb",
            linewidth=0.7,
            alpha=0.85,
        ),
        zorder=5,
        linespacing=1.3,
    )


def draw_input_arrow(ax, stage):
    """Draw an input arrow to the left of stage 0."""
    x0 = box_x(0)
    y = Y_CENTER
    ax.annotate(
        "",
        xy=(x0, y),
        xytext=(x0 - 0.55, y),
        arrowprops=dict(
            arrowstyle="->,head_width=0.22,head_length=0.15",
            color="#555555",
            lw=1.4,
        ),
        zorder=4,
    )
    ax.text(
        x0 - 0.57, y + 0.28,
        stage["in_shape"],
        ha="center", va="bottom",
        fontsize=7, color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="#f5f5f5",
            edgecolor="#cccccc",
            linewidth=0.7,
        ),
        zorder=5,
        linespacing=1.3,
    )


def draw_output_label(ax, stage):
    """Draw output label to the right of the last stage."""
    i = N_STAGES - 1
    x_end = box_x(i) + STAGE_W
    y = Y_CENTER
    ax.annotate(
        "",
        xy=(x_end + 0.55, y),
        xytext=(x_end, y),
        arrowprops=dict(
            arrowstyle="->,head_width=0.22,head_length=0.15",
            color="#555555",
            lw=1.4,
        ),
        zorder=4,
    )
    ax.text(
        x_end + 0.57, y + 0.28,
        stage["out_shape"],
        ha="center", va="bottom",
        fontsize=7, color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="#f5f5f5",
            edgecolor="#cccccc",
            linewidth=0.7,
        ),
        zorder=5,
        linespacing=1.3,
    )


def add_model_braces(ax):
    """Add curly-brace annotations grouping Vision Encoder and VLM blocks."""
    brace_y = Y_CENTER - STAGE_H / 2.0 - 0.55

    # Vision Encoder spans stages 0–2
    x_start_ve = box_x(0)
    x_end_ve = box_x(2) + STAGE_W
    mid_ve = (x_start_ve + x_end_ve) / 2.0

    ax.annotate(
        "",
        xy=(x_start_ve, brace_y + 0.12),
        xytext=(x_start_ve, brace_y),
        arrowprops=dict(arrowstyle="-", color="#4E79A7", lw=1.2),
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(x_end_ve, brace_y + 0.12),
        xytext=(x_end_ve, brace_y),
        arrowprops=dict(arrowstyle="-", color="#4E79A7", lw=1.2),
        zorder=3,
    )
    ax.plot([x_start_ve, x_end_ve], [brace_y, brace_y],
            color="#4E79A7", lw=1.2, zorder=3)
    ax.text(mid_ve, brace_y - 0.18,
            "Vision Encoder (SigLIP-400M backbone)",
            ha="center", va="top", fontsize=8, color="#4E79A7", fontstyle="italic")

    # VLM spans stage 3
    x_start_vlm = box_x(3)
    x_end_vlm = box_x(3) + STAGE_W
    mid_vlm = (x_start_vlm + x_end_vlm) / 2.0

    ax.plot([x_start_vlm, x_end_vlm], [brace_y, brace_y],
            color="#76B7B2", lw=1.2, zorder=3)
    ax.annotate(
        "",
        xy=(x_start_vlm, brace_y + 0.12),
        xytext=(x_start_vlm, brace_y),
        arrowprops=dict(arrowstyle="-", color="#76B7B2", lw=1.2),
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(x_end_vlm, brace_y + 0.12),
        xytext=(x_end_vlm, brace_y),
        arrowprops=dict(arrowstyle="-", color="#76B7B2", lw=1.2),
        zorder=3,
    )
    ax.text(mid_vlm, brace_y - 0.18,
            "Qwen3-VL-8B",
            ha="center", va="top", fontsize=8, color="#76B7B2", fontstyle="italic")

    # Expert Decoder spans stages 4–6
    x_start_ed = box_x(4)
    x_end_ed = box_x(6) + STAGE_W
    mid_ed = (x_start_ed + x_end_ed) / 2.0

    ax.plot([x_start_ed, x_end_ed], [brace_y, brace_y],
            color="#59A14F", lw=1.2, zorder=3)
    ax.annotate(
        "",
        xy=(x_start_ed, brace_y + 0.12),
        xytext=(x_start_ed, brace_y),
        arrowprops=dict(arrowstyle="-", color="#59A14F", lw=1.2),
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(x_end_ed, brace_y + 0.12),
        xytext=(x_end_ed, brace_y),
        arrowprops=dict(arrowstyle="-", color="#59A14F", lw=1.2),
        zorder=3,
    )
    ax.text(mid_ed, brace_y - 0.18,
            "Expert Decoder + Diffusion Head",
            ha="center", va="top", fontsize=8, color="#59A14F", fontstyle="italic")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F8F8")

    # Title
    ax.text(
        FIG_W / 2, FIG_H - 0.45,
        "Alpamayo-R1-10B  —  7-Stage Inference Pipeline",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#1a1a1a",
    )
    ax.text(
        FIG_W / 2, FIG_H - 0.88,
        "Vision Encoder (SigLIP)  →  VLM (Qwen3-VL-8B)  →  Expert Decoder  →  Flow Matching Diffusion",
        ha="center", va="center",
        fontsize=9, color="#555555",
    )

    # Draw boxes
    for i, stage in enumerate(STAGES):
        draw_stage_box(ax, i, stage)

    # Draw inter-stage arrows with tensor shape labels
    for i in range(N_STAGES - 1):
        draw_arrow_with_labels(ax, i, STAGES[i], STAGES[i + 1])

    # Input and output peripheral arrows
    draw_input_arrow(ax, STAGES[0])
    draw_output_label(ax, STAGES[-1])

    # Component grouping annotations
    add_model_braces(ax)

    # Legend for color mapping
    legend_elements = [
        mpatches.Patch(facecolor=s["color"], label=s["title"].replace("\n", " "))
        for s in STAGES
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=N_STAGES,
        fontsize=7.5,
        framealpha=0.85,
        edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(pad=0.5)

    out_path = "/home/seungwoo/workspace/analysis/figures/pipeline_overview.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
