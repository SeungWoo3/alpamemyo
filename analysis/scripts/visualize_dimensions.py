"""
Alpamayo-R1-10B Tensor Dimension Changes Visualization
Grouped bar chart (log scale) showing sequence length and feature dimension
at each of the 7 pipeline stages.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


# ---------------------------------------------------------------------------
# Pipeline stage data
# ---------------------------------------------------------------------------
# Assume a representative input of 448×448 pixels → N = (448/16)^2 = 784 patches
# After 2×2 merge:  N/4 = 196  (spatial tokens fed to VLM)
# VLM output "S" treated as 196 (same spatial context, no generation growth shown)
# Diffusion / action / trajectory are over the 64-waypoint action space.

STAGES = [
    {
        "label": "1. Patch\nEmbedding",
        "seq_len": 784,      # N patches (448×448, patch=16)
        "feat_dim": 1152,    # Vision Encoder hidden dim
        "note": "N=784\nd=1152",
    },
    {
        "label": "2. Vision\nTransformer\n(27 layers)",
        "seq_len": 784,
        "feat_dim": 1152,
        "note": "N=784\nd=1152",
    },
    {
        "label": "3. Patch\nMerger",
        "seq_len": 196,      # 784 / 4
        "feat_dim": 4096,    # projected to VLM hidden dim
        "note": "N/4=196\nd=4096",
    },
    {
        "label": "4. VLM\nGeneration\n(36 layers)",
        "seq_len": 196,      # spatial tokens
        "feat_dim": 4096,    # VLM hidden dim
        "note": "S=196\nd=4096",
    },
    {
        "label": "5. Diffusion\nSampling\n(10 steps)",
        "seq_len": 64,       # waypoints
        "feat_dim": 2,       # accel + curvature
        "note": "W=64\nd=2",
    },
    {
        "label": "6. Action →\nTrajectory",
        "seq_len": 64,
        "feat_dim": 3,       # x, y, theta
        "note": "W=64\nd=3",
    },
    {
        "label": "7. Final\nOutput",
        "seq_len": 64,
        "feat_dim": 3,
        "note": "W=64\nd=3",
    },
]

N_STAGES = len(STAGES)
STAGE_LABELS = [s["label"] for s in STAGES]
SEQ_LENS = np.array([s["seq_len"] for s in STAGES], dtype=float)
FEAT_DIMS = np.array([s["feat_dim"] for s in STAGES], dtype=float)
TOTAL_ELEMS = SEQ_LENS * FEAT_DIMS   # total number of elements per tensor

# Colors
COLOR_SEQ = "#4E79A7"    # blue
COLOR_FEAT = "#F28E2B"   # orange
COLOR_TOTAL = "#E15759"  # red (line)


# ---------------------------------------------------------------------------
# Figure layout: 2 rows
#   Top row: grouped bar chart (log scale) — seq_len vs feat_dim
#   Bottom row: total tensor elements (log scale)
# ---------------------------------------------------------------------------

def add_bar_labels(ax, rects, values, fmt="{:.0f}", offset_frac=0.08):
    """Place value labels above bars."""
    for rect, val in zip(rects, values):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2.0
        # position label just above the bar in log-space
        y_log = np.log10(h) + offset_frac * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(
            x, h * 1.25,
            fmt.format(int(val)),
            ha="center", va="bottom",
            fontsize=7.5, color="#333333", fontweight="bold",
        )


def draw_stage_annotations(ax, stages, y_anchor, log_scale=True):
    """Draw small note boxes below each stage cluster."""
    xlocs = ax.get_xticks()
    for x, stage in zip(xlocs, stages):
        ax.text(
            x, y_anchor,
            stage["note"],
            ha="center", va="top",
            fontsize=7, color="#555555",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                linewidth=0.6,
            ),
        )


def main():
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#F8F8F8")

    gs = fig.add_gridspec(
        2, 1,
        hspace=0.52,
        top=0.88, bottom=0.13,
        left=0.07, right=0.97,
    )

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    x = np.arange(N_STAGES)
    bar_w = 0.35

    # -----------------------------------------------------------------------
    # Top panel: Sequence Length vs Feature Dimension (grouped bars, log)
    # -----------------------------------------------------------------------
    bars_seq = ax_top.bar(
        x - bar_w / 2, SEQ_LENS,
        width=bar_w,
        color=COLOR_SEQ, alpha=0.88,
        label="Sequence Length (tokens / waypoints)",
        zorder=3,
    )
    bars_feat = ax_top.bar(
        x + bar_w / 2, FEAT_DIMS,
        width=bar_w,
        color=COLOR_FEAT, alpha=0.88,
        label="Feature Dimension",
        zorder=3,
    )

    ax_top.set_yscale("log")
    ax_top.set_ylim(0.8, 20000)

    # Gridlines
    ax_top.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_top.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
                color="#cccccc", alpha=0.8, zorder=0)
    ax_top.grid(axis="y", which="minor", linestyle=":", linewidth=0.4,
                color="#dddddd", alpha=0.5, zorder=0)
    ax_top.set_axisbelow(True)

    # Bar value labels
    for rect, val in zip(bars_seq, SEQ_LENS):
        h = rect.get_height()
        ax_top.text(
            rect.get_x() + rect.get_width() / 2.0,
            h * 1.35,
            f"{int(val):,}",
            ha="center", va="bottom",
            fontsize=7, color=COLOR_SEQ, fontweight="bold",
        )
    for rect, val in zip(bars_feat, FEAT_DIMS):
        h = rect.get_height()
        ax_top.text(
            rect.get_x() + rect.get_width() / 2.0,
            h * 1.35,
            f"{int(val):,}",
            ha="center", va="bottom",
            fontsize=7, color=COLOR_FEAT, fontweight="bold",
        )

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(STAGE_LABELS, fontsize=8.5, linespacing=1.3)
    ax_top.set_ylabel("Dimension Size (log scale)", fontsize=9)
    ax_top.set_title(
        "Sequence Length vs Feature Dimension Across Pipeline Stages",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_top.spines[["top", "right"]].set_visible(False)
    ax_top.legend(
        fontsize=8.5, loc="upper right",
        framealpha=0.85, edgecolor="#cccccc",
    )

    # Highlight key transitions with vertical dashed lines between groups
    transition_points = [2.5, 3.5, 4.5]   # between stages 3-4, 4-5, 5-6
    transition_labels = [
        "Merger\n(4096-proj)",
        "VLM →\nDiffusion",
        "Action →\nTraj",
    ]
    for xp, lbl in zip(transition_points, transition_labels):
        ax_top.axvline(xp, color="#999999", linestyle="--", linewidth=0.8, alpha=0.6)

    # -----------------------------------------------------------------------
    # Bottom panel: Total tensor elements (seq_len × feat_dim)
    # -----------------------------------------------------------------------
    bars_total = ax_bot.bar(
        x, TOTAL_ELEMS,
        width=0.55,
        color=[
            "#4E79A7", "#4E79A7",   # Vision Encoder stages
            "#E15759",               # Patch Merger (spike)
            "#76B7B2",               # VLM
            "#59A14F", "#59A14F",   # Expert Decoder / Diffusion
            "#B07AA1",               # Final Output
        ],
        alpha=0.88,
        zorder=3,
    )

    # Overlay line
    ax_bot.plot(
        x, TOTAL_ELEMS,
        color="#333333", linewidth=1.4, marker="o",
        markersize=5, zorder=4, label="Total elements",
    )

    ax_bot.set_yscale("log")
    ax_bot.set_ylim(1, 2e7)
    ax_bot.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )
    ax_bot.grid(axis="y", which="major", linestyle="--", linewidth=0.6,
                color="#cccccc", alpha=0.8, zorder=0)
    ax_bot.grid(axis="y", which="minor", linestyle=":", linewidth=0.4,
                color="#dddddd", alpha=0.5, zorder=0)
    ax_bot.set_axisbelow(True)

    # Bar value labels
    for rect, val in zip(bars_total, TOTAL_ELEMS):
        h = rect.get_height()
        label = f"{int(val):,}" if val >= 1000 else f"{int(val)}"
        ax_bot.text(
            rect.get_x() + rect.get_width() / 2.0,
            h * 1.5,
            label,
            ha="center", va="bottom",
            fontsize=7, color="#222222", fontweight="bold",
        )

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(STAGE_LABELS, fontsize=8.5, linespacing=1.3)
    ax_bot.set_ylabel("Total Elements = seq × feat (log scale)", fontsize=9)
    ax_bot.set_title(
        "Total Tensor Elements (Sequence Length × Feature Dim) per Stage",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax_bot.spines[["top", "right"]].set_visible(False)

    # Color legend for component groups
    legend_patches = [
        mpatches.Patch(facecolor="#4E79A7", alpha=0.88, label="Vision Encoder (stages 1-2)"),
        mpatches.Patch(facecolor="#E15759", alpha=0.88, label="Patch Merger (stage 3)"),
        mpatches.Patch(facecolor="#76B7B2", alpha=0.88, label="VLM Generation (stage 4)"),
        mpatches.Patch(facecolor="#59A14F", alpha=0.88, label="Diffusion / Action (stages 5-6)"),
        mpatches.Patch(facecolor="#B07AA1", alpha=0.88, label="Final Output (stage 7)"),
    ]
    ax_bot.legend(
        handles=legend_patches,
        fontsize=7.5, loc="upper right",
        framealpha=0.85, edgecolor="#cccccc",
    )

    # -----------------------------------------------------------------------
    # Figure-level title and footnote
    # -----------------------------------------------------------------------
    fig.suptitle(
        "Alpamayo-R1-10B  —  Tensor Dimension Changes Through Pipeline",
        fontsize=13, fontweight="bold", y=0.96, color="#1a1a1a",
    )
    fig.text(
        0.5, 0.005,
        "Assumptions: input 448×448 px → N=784 patches (patch=16); "
        "after 2×2 merge N/4=196; action space W=64 waypoints.",
        ha="center", va="bottom",
        fontsize=7.5, color="#777777", style="italic",
    )

    out_path = "/home/seungwoo/workspace/analysis/figures/dimension_changes.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
