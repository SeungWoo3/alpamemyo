"""
visualize_diffusion.py

Flow Matching diffusion process visualization for Alpamayo-R1-10B.

Panels
------
Top    : Horizontal 10-step Euler flow diagram (t=0 → t=1)
Middle : 2D conceptual denoising plot (5 intermediate states)
Bottom : Mathematical formulation box

Source reference
----------------
alpamayo/src/alpamayo_r1/diffusion/flow_matching.py
  - time_steps = linspace(0, 1, 11)   [size 11, 10 intervals]
  - x = randn(B, 64, 2)               initial Gaussian noise
  - for i in range(10):
        dt = time_steps[i+1] - time_steps[i]   # 0.1
        v  = step_fn(x, t_i)
        x  = x + dt * v

Saved to: /home/seungwoo/workspace/analysis/figures/diffusion_steps.png
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG       = "#F8FAFC"
C_NOISE    = "#C0392B"      # deep red   — t=0 (noise start)
C_DATA     = "#1A7A4A"      # deep green — t=1 (data)
C_ARROW    = "#2C3E50"
C_VEL      = "#7D3C98"      # purple     — velocity v
C_STEP_HI  = "#1565C0"      # highlight border for step boxes
C_COND     = "#D35400"      # orange     — Expert Decoder label
C_GRID     = "#D5D8DC"
C_DIM      = "#7F8C8D"
C_MATH_BG  = "#EBF5FB"
C_WHITE    = "#FFFFFF"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _lerp_color(t: float):
    """Interpolate hex colour: t=0 → red (noise), t=1 → green (data)."""
    r0, g0, b0 = 0xC0 / 255, 0x39 / 255, 0x2B / 255   # noise colour
    r1, g1, b1 = 0x1A / 255, 0x7A / 255, 0x4A / 255   # data colour
    return (r0 + (r1 - r0) * t,
            g0 + (g1 - g0) * t,
            b0 + (b1 - b0) * t)


def _rounded_box(ax, cx, cy, w, h, label, sublabel=None,
                 fc=C_WHITE, ec=C_ARROW, lw=1.6, fs=9, sub_fs=7.5,
                 tc=C_ARROW, bold=False, rounding=0.04, zorder=4):
    """Draw a rounded-rectangle block centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ty = cy + h * 0.12 if sublabel else cy
    ax.text(cx, ty, label, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, zorder=zorder + 1)
    if sublabel:
        ax.text(cx, cy - h * 0.20, sublabel, ha="center", va="center",
                fontsize=sub_fs, color=C_DIM, style="italic",
                zorder=zorder + 1)


def _harrow(ax, x0, x1, y, color=C_ARROW, lw=1.5, label=None, label_y_off=0.12):
    """Horizontal annotated arrow."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12),
                zorder=5)
    if label:
        ax.text((x0 + x1) / 2, y + label_y_off, label,
                ha="center", va="bottom", fontsize=6.5,
                color=color, style="italic", zorder=6)


# ---------------------------------------------------------------------------
# Panel 1 — Horizontal step-flow diagram
# ---------------------------------------------------------------------------
def draw_step_flow(ax):
    """
    Euler integration chain:
      x_0 (noise) --[v_0]--> x_1 --[v_1]--> ... --[v_9]--> x_1.0 (data)
    where subscripts refer to the *diffusion* timestep index (0..10),
    and the floating-point flow time t runs 0.0, 0.1, ..., 1.0.
    """
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(8, 5.25,
            "Flow Matching: 10-Step Euler Integration  "
            r"(t: 0.0 $\to$ 1.0, noise $\to$ data)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=C_ARROW)

    # ---- geometry ----
    N_STEPS = 11          # 11 state boxes (t = 0.0 … 1.0)
    TIMES   = np.linspace(0.0, 1.0, N_STEPS)
    BOX_W   = 1.0
    BOX_H   = 2.4
    X_MARGIN = 0.55
    X_TOTAL  = 15.4
    X_STEP   = (X_TOTAL - X_MARGIN) / (N_STEPS - 1)
    Y_BOX    = 2.8

    centres = [X_MARGIN + i * X_STEP for i in range(N_STEPS)]

    # ---- state boxes ----
    for i, t in enumerate(TIMES):
        fc = _lerp_color(t)
        is_first = (i == 0)
        is_last  = (i == N_STEPS - 1)
        ec_hex   = C_NOISE if is_first else (C_DATA if is_last else C_STEP_HI)
        lw       = 2.2 if (is_first or is_last) else 1.4
        lbl      = r"$x_0$" if is_first else (r"$x_1$" if is_last else r"$x_{t}$")
        sub      = f"t = {t:.1f}"
        _rounded_box(ax, centres[i], Y_BOX, BOX_W * 0.90, BOX_H,
                     lbl, sublabel=sub,
                     fc=fc, ec=ec_hex, lw=lw,
                     fs=10, sub_fs=8, tc=C_WHITE, bold=True,
                     rounding=0.06, zorder=4)

    # ---- arrows between boxes ----
    for i in range(N_STEPS - 1):
        x0 = centres[i]     + BOX_W * 0.90 / 2
        x1 = centres[i + 1] - BOX_W * 0.90 / 2
        t_label = f"t={TIMES[i]:.1f}"
        _harrow(ax, x0, x1, Y_BOX,
                color=C_VEL, lw=1.8,
                label=r"$+\,dt\cdot v$",
                label_y_off=0.18)

    # ---- Expert Decoder conditioning band ----
    Y_COND = 4.75
    cond_bg = FancyBboxPatch(
        (centres[0] - 0.3, Y_COND - 0.28), centres[-1] - centres[0] + 0.6, 0.52,
        boxstyle="round,pad=0,rounding_size=0.05",
        facecolor="#FEF9E7", edgecolor=C_COND, linewidth=1.3, zorder=2, alpha=0.9,
    )
    ax.add_patch(cond_bg)
    ax.text(8, Y_COND,
            "Expert Decoder (16 layers, KV cache)  ·  conditions each step_fn call",
            ha="center", va="center", fontsize=8.5,
            color=C_COND, fontweight="bold", zorder=3)

    # dashed drop-lines from conditioning bar to each box top
    for cx in centres:
        ax.plot([cx, cx],
                [Y_COND - 0.28, Y_BOX + BOX_H / 2],
                color=C_COND, lw=0.9, linestyle="--", alpha=0.5, zorder=2)

    # ---- side labels ----
    ax.text(centres[0], Y_BOX - BOX_H / 2 - 0.38,
            "Gaussian Noise\n" r"$x_0 \sim \mathcal{N}(0,I)$",
            ha="center", va="top", fontsize=8, color=C_NOISE, fontweight="bold")
    ax.text(centres[-1], Y_BOX - BOX_H / 2 - 0.38,
            "Denoised Actions\n(accel, curvature)",
            ha="center", va="top", fontsize=8, color=C_DATA, fontweight="bold")

    # ---- panel label ----
    ax.text(-0.02, 1.02, "Top", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=C_ARROW,
            va="top", ha="left")


# ---------------------------------------------------------------------------
# Panel 2 — Conceptual 2D denoising plot (5 snapshots)
# ---------------------------------------------------------------------------
def draw_denoising_plot(ax):
    """
    Show 5 intermediate states (t=0, 0.2, 0.5, 0.8, 1.0) of an action
    sequence being refined from random scatter toward a smooth sinusoidal curve.

    The action space is 2D (accel, curvature) across 64 waypoints.
    We reduce to 1D here: waypoint index on x-axis, action value on y-axis,
    and overlay the 5 states.
    """
    rng = np.random.default_rng(7)
    N = 64
    wp = np.arange(N)

    # Ground-truth clean trajectory in action space (sinusoidal)
    clean = 0.6 * np.sin(2 * np.pi * wp / 32) + 0.2 * np.cos(2 * np.pi * wp / 20)

    # Pure noise
    noise = rng.standard_normal(N) * 1.5

    # 5 interpolated states
    t_show   = [0.0, 0.2, 0.5, 0.8, 1.0]
    cmap     = plt.cm.RdYlGn                  # red → green
    alphas   = [0.55, 0.60, 0.70, 0.80, 1.00]
    lws      = [0.8,  1.0,  1.2,  1.5,  2.0]

    for idx, t in enumerate(t_show):
        state = (1.0 - t) * noise + t * clean  # linear interpolation
        # add diminishing jitter to simulate imperfect step_fn
        jitter_scale = (1.0 - t) * 0.25
        state += rng.standard_normal(N) * jitter_scale
        color = cmap(t)
        ax.plot(wp, state,
                color=color, lw=lws[idx], alpha=alphas[idx],
                label=f"t = {t:.1f}", zorder=3 + idx)

    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 63)
    ax.set_xlabel("Waypoint index  (0 … 63)", fontsize=9, color=C_DIM)
    ax.set_ylabel("Action value  (normalised)", fontsize=9, color=C_DIM)
    ax.set_title(
        "Conceptual Trajectory Refinement in Action Space\n"
        r"(accel / curvature over 64 waypoints, 5 snapshots)",
        fontsize=10, fontweight="bold", pad=8, color=C_ARROW,
    )
    ax.tick_params(labelsize=8, colors=C_DIM)
    ax.grid(True, color=C_GRID, lw=0.6, linestyle="--", alpha=0.7)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)

    # colourbar-style legend
    leg = ax.legend(fontsize=8.5, loc="upper right",
                    framealpha=0.9, edgecolor=C_GRID,
                    title="Flow time t", title_fontsize=8.5)
    leg.get_title().set_color(C_DIM)

    # annotations
    ax.text(1, noise[0] * 0.9,
            r"$x_0$ (noise)", fontsize=8, color=C_NOISE, fontweight="bold")
    ax.text(1, clean[0] + 0.15,
            r"$x_1$ (data)", fontsize=8, color=C_DATA, fontweight="bold")

    ax.text(-0.08, 1.04, "Middle", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=C_ARROW,
            va="top", ha="left")


# ---------------------------------------------------------------------------
# Panel 3 — Mathematical formulation box
# ---------------------------------------------------------------------------
def draw_math_box(ax):
    """
    Render the key equations and pseudocode as formatted text inside a
    styled box.
    """
    ax.set_facecolor(C_MATH_BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Outer border
    border = FancyBboxPatch(
        (0.08, 0.08), 9.84, 4.84,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=C_MATH_BG, edgecolor=C_STEP_HI, linewidth=2.0, zorder=1,
    )
    ax.add_patch(border)

    ax.text(5, 4.55,
            "Mathematical Formulation  —  Flow Matching (Euler)",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=C_STEP_HI, zorder=2)

    lines = [
        # (x, y, text, fontsize, color, ha, style, weight)
        (0.45, 3.95,
         r"Initialisation:   $x_0 \;\sim\; \mathcal{N}(0,\,I),$"
         r"   $\mathrm{shape}\;(B,\;64,\;2)$",
         10, C_ARROW, "left", "normal", "normal"),

        (0.45, 3.35,
         r"Timesteps:   $\{t_i\}_{i=0}^{10} = \mathrm{linspace}(0,\,1,\,11)$",
         10, C_ARROW, "left", "normal", "normal"),

        (0.45, 2.75,
         r"Euler loop   ($i = 0, 1, \ldots, 9$):",
         10, C_ARROW, "left", "normal", "bold"),

        (1.0, 2.20,
         r"$dt \;=\; 0.1$",
         10, C_VEL, "left", "italic", "normal"),

        (1.0, 1.70,
         r"$v \;=\; \mathrm{step\_fn}(x,\;t_i)$",
         10, C_VEL, "left", "italic", "normal"),

        (1.0, 1.20,
         r"$x \;\leftarrow\; x \;+\; dt \cdot v$",
         10.5, C_VEL, "left", "italic", "bold"),

        (0.45, 0.60,
         r"step\_fn:   FourierEncode$(x, t)$"
         r"  $\rightarrow$ MLP $\rightarrow$  Expert Decoder (16 layers, KV cache)"
         r"  $\rightarrow$ Linear $\rightarrow$  $v \in \mathbb{R}^{B \times 64 \times 2}$",
         8.8, C_COND, "left", "normal", "normal"),
    ]

    for (lx, ly, txt, fs, col, ha, style, weight) in lines:
        ax.text(lx, ly, txt,
                ha=ha, va="center",
                fontsize=fs, color=col,
                fontstyle=style, fontweight=weight, zorder=2)

    # Divider line above step_fn description
    ax.plot([0.3, 9.7], [0.93, 0.93], color=C_GRID, lw=1.0, zorder=2)

    ax.text(-0.02, 1.02, "Bottom", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=C_ARROW,
            va="top", ha="left")


# ---------------------------------------------------------------------------
# Assemble figure
# ---------------------------------------------------------------------------
def build_figure() -> plt.Figure:
    fig = plt.figure(figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor(C_BG)

    fig.suptitle(
        "Alpamayo-R1-10B  |  Flow Matching Diffusion Process",
        fontsize=14, fontweight="bold", color=C_ARROW, y=0.99,
    )

    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.15, 1.10, 0.80],
        hspace=0.42,
        left=0.06, right=0.97,
        top=0.96, bottom=0.04,
    )

    ax_top    = fig.add_subplot(gs[0])
    ax_mid    = fig.add_subplot(gs[1])
    ax_bot    = fig.add_subplot(gs[2])

    draw_step_flow(ax_top)
    draw_denoising_plot(ax_mid)
    draw_math_box(ax_bot)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_path = "/home/seungwoo/workspace/analysis/figures/diffusion_steps.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = build_figure()
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=C_BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
