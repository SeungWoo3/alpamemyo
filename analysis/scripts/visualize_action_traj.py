"""
visualize_action_traj.py

Visualizes the Alpamayo-R1-10B action-to-trajectory conversion pipeline.
Unicycle model:
    x_{k+1}   = x_k   + v_k * cos(yaw_k) * dt
    y_{k+1}   = y_k   + v_k * sin(yaw_k) * dt
    yaw_{k+1} = yaw_k + v_k * curvature_k * dt

accel  -> velocity via cumulative integration (v_k = v_{k-1} + a_k * dt)
curvature -> yaw  via integration with velocity weighting
Output: (64, 3) = [x, y, yaw] per waypoint
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Colour palette ──────────────────────────────────────────────────────────
C_ACCEL  = "#E07B54"   # warm orange
C_CURV   = "#5B8DB8"   # steel blue
C_VEL    = "#E07B54"
C_YAW    = "#5B8DB8"
C_TRAJ   = "#2E7D32"   # forest green
C_ARROW  = "#7B1FA2"   # purple
C_GRID   = "#DDDDDD"

# ── Synthetic action generation ─────────────────────────────────────────────
def make_synthetic_actions(n: int = 64, dt: float = 0.1, seed: int = 42) -> dict:
    """
    Generate a smooth, illustrative action sequence of length n.
    Returns a dict with all integrated quantities.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt                        # time axis [s]

    # -- Acceleration profile: smooth sinusoidal + slight deceleration ramp
    accel = (
        0.8 * np.sin(2 * np.pi * t / (n * dt) * 1.5)
        + 0.3 * np.cos(2 * np.pi * t / (n * dt) * 3.5)
        + 0.05 * rng.standard_normal(n)
    )
    accel = np.clip(accel, -2.0, 2.0)

    # -- Curvature profile: slow sinusoid (lazy S-curve)
    curvature = (
        0.15 * np.sin(2 * np.pi * t / (n * dt) * 0.8 + 0.4)
        + 0.05 * np.cos(2 * np.pi * t / (n * dt) * 2.0)
        + 0.01 * rng.standard_normal(n)
    )
    curvature = np.clip(curvature, -0.5, 0.5)

    # -- Velocity integration (v_0 = 3 m/s initial)
    v0 = 3.0
    velocity = np.empty(n)
    velocity[0] = v0
    for k in range(1, n):
        velocity[k] = max(0.0, velocity[k - 1] + accel[k] * dt)

    # -- Yaw integration (yaw_0 = 0 rad)
    yaw = np.empty(n)
    yaw[0] = 0.0
    for k in range(1, n):
        yaw[k] = yaw[k - 1] + velocity[k - 1] * curvature[k - 1] * dt

    # -- Position integration (x_0=y_0=0)
    xs = np.empty(n)
    ys = np.empty(n)
    xs[0] = 0.0
    ys[0] = 0.0
    for k in range(1, n):
        xs[k] = xs[k - 1] + velocity[k - 1] * np.cos(yaw[k - 1]) * dt
        ys[k] = ys[k - 1] + velocity[k - 1] * np.sin(yaw[k - 1]) * dt

    return dict(t=t, accel=accel, curvature=curvature,
                velocity=velocity, yaw=yaw, x=xs, y=ys)


# ── Annotation helpers ──────────────────────────────────────────────────────
def add_equation_box(ax, text, xy, fontsize=9.5, color="white", bg="#37474F", pad=0.5):
    ax.annotate(
        text, xy=xy, xycoords="axes fraction",
        fontsize=fontsize, color=color,
        ha="center", va="center",
        bbox=dict(boxstyle=f"round,pad={pad}", fc=bg, ec="none", alpha=0.90),
    )


# ── Main plot ───────────────────────────────────────────────────────────────
def plot_action_trajectory(data: dict, save_path: str) -> None:
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor("#FAFAFA")

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.48,
        wspace=0.38,
        left=0.09, right=0.97,
        top=0.93, bottom=0.07,
    )

    t        = data["t"]
    step_idx = np.arange(len(t))

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Alpamayo-R1-10B  ·  Action Space → Trajectory (Unicycle Model)",
        fontsize=15, fontweight="bold", y=0.975,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 0 — Raw action outputs (accel & curvature)
    # ═══════════════════════════════════════════════════════════════════════
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_cur = fig.add_subplot(gs[0, 1])

    # Acceleration
    ax_acc.set_facecolor("#F5F5F5")
    ax_acc.plot(step_idx, data["accel"], color=C_ACCEL, lw=2.0, label="Acceleration")
    ax_acc.axhline(0, color="#999999", lw=0.8, ls="--")
    ax_acc.fill_between(step_idx, data["accel"], 0,
                        where=data["accel"] >= 0,
                        alpha=0.18, color=C_ACCEL, label="+accel")
    ax_acc.fill_between(step_idx, data["accel"], 0,
                        where=data["accel"] < 0,
                        alpha=0.18, color="#5B8DB8", label="-accel")
    ax_acc.set_xlabel("Waypoint index  (0–63)")
    ax_acc.set_ylabel("Acceleration  [m/s²]")
    ax_acc.set_title("Action Dimension 1 — Acceleration", pad=6)
    ax_acc.legend(fontsize=9, loc="upper right")
    ax_acc.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_acc.grid(True, color=C_GRID, lw=0.6)
    ax_acc.grid(True, which="minor", color=C_GRID, lw=0.3, ls=":")

    add_equation_box(ax_acc,
        r"$a_k \in [-2, 2]\ \mathrm{m/s^2}$" "\n" r"dim 0 of 64×2 output",
        xy=(0.35, 0.14), fontsize=9)

    # Curvature
    ax_cur.set_facecolor("#F5F5F5")
    ax_cur.plot(step_idx, data["curvature"], color=C_CURV, lw=2.0)
    ax_cur.axhline(0, color="#999999", lw=0.8, ls="--")
    ax_cur.fill_between(step_idx, data["curvature"], 0,
                        alpha=0.18, color=C_CURV)
    ax_cur.set_xlabel("Waypoint index  (0–63)")
    ax_cur.set_ylabel("Curvature  [1/m]")
    ax_cur.set_title("Action Dimension 2 — Curvature", pad=6)
    ax_cur.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_cur.grid(True, color=C_GRID, lw=0.6)
    ax_cur.grid(True, which="minor", color=C_GRID, lw=0.3, ls=":")

    add_equation_box(ax_cur,
        r"$\kappa_k \in [-0.5, 0.5]\ \mathrm{m^{-1}}$" "\n" r"dim 1 of 64×2 output",
        xy=(0.35, 0.14), fontsize=9)

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 1 — Integration results (velocity & yaw)
    # ═══════════════════════════════════════════════════════════════════════
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_yaw = fig.add_subplot(gs[1, 1])

    # Velocity
    ax_vel.set_facecolor("#F5F5F5")
    ax_vel.plot(step_idx, data["velocity"], color=C_VEL, lw=2.2)
    ax_vel.fill_between(step_idx, data["velocity"], alpha=0.15, color=C_VEL)
    ax_vel.set_xlabel("Waypoint index  (0–63)")
    ax_vel.set_ylabel("Velocity  [m/s]")
    ax_vel.set_title("Integrated Velocity  (cumulative accel)", pad=6)
    ax_vel.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_vel.grid(True, color=C_GRID, lw=0.6)
    ax_vel.grid(True, which="minor", color=C_GRID, lw=0.3, ls=":")

    add_equation_box(ax_vel,
        r"$v_k = v_{k-1} + a_k \cdot \Delta t$" "\n" r"$\Delta t = 0.1\,\mathrm{s}$",
        xy=(0.72, 0.82), fontsize=9)

    # Yaw
    ax_yaw.set_facecolor("#F5F5F5")
    ax_yaw.plot(step_idx, np.degrees(data["yaw"]), color=C_YAW, lw=2.2)
    ax_yaw.fill_between(step_idx, np.degrees(data["yaw"]), alpha=0.15, color=C_YAW)
    ax_yaw.set_xlabel("Waypoint index  (0–63)")
    ax_yaw.set_ylabel("Yaw angle  [deg]")
    ax_yaw.set_title("Integrated Yaw  (curvature × velocity)", pad=6)
    ax_yaw.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_yaw.grid(True, color=C_GRID, lw=0.6)
    ax_yaw.grid(True, which="minor", color=C_GRID, lw=0.3, ls=":")

    add_equation_box(ax_yaw,
        r"$\psi_k = \psi_{k-1} + v_{k-1} \cdot \kappa_{k-1} \cdot \Delta t$",
        xy=(0.50, 0.14), fontsize=9)

    # ═══════════════════════════════════════════════════════════════════════
    # ROW 2 — 2D trajectory (full-width)
    # ═══════════════════════════════════════════════════════════════════════
    ax_traj = fig.add_subplot(gs[2, :])   # span both columns
    ax_traj.set_facecolor("#F0F4F0")

    x, y, yaw_rad = data["x"], data["y"], data["yaw"]

    # Colourmap by progress
    n = len(x)
    cmap_traj = plt.cm.viridis
    for i in range(n - 1):
        c = cmap_traj(i / n)
        ax_traj.plot(x[i:i+2], y[i:i+2], color=c, lw=2.5, solid_capstyle="round")

    # Start / end markers
    ax_traj.scatter(x[0], y[0], s=120, color="#1976D2", zorder=5,
                    label="Start", edgecolors="white", linewidths=1.5)
    ax_traj.scatter(x[-1], y[-1], s=120, color="#D32F2F", zorder=5,
                    label="End", marker="*", edgecolors="white", linewidths=1.0)

    # Yaw arrows at every 8th waypoint
    arrow_indices = list(range(0, n, 8))
    arrow_len = 0.4
    for idx in arrow_indices:
        dx = arrow_len * np.cos(yaw_rad[idx])
        dy = arrow_len * np.sin(yaw_rad[idx])
        ax_traj.annotate(
            "", xy=(x[idx] + dx, y[idx] + dy),
            xytext=(x[idx], y[idx]),
            arrowprops=dict(
                arrowstyle="-|>", color=C_ARROW,
                lw=1.5, mutation_scale=12,
            ),
        )

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap_traj,
                               norm=plt.Normalize(vmin=0, vmax=(n-1)*0.1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_traj, pad=0.02, shrink=0.85)
    cbar.set_label("Time  [s]", fontsize=10)

    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x  [m]")
    ax_traj.set_ylabel("y  [m]")
    ax_traj.set_title(
        "2D Trajectory — Unicycle Model Integration  "
        r"($64 \times 2$ actions $\rightarrow$ $64 \times 3$ waypoints $[x, y, \psi]$)",
        pad=6,
    )
    ax_traj.legend(fontsize=9, loc="upper left")
    ax_traj.grid(True, color=C_GRID, lw=0.6)

    # Equation annotation bottom-right
    eq_text = (
        r"$x_{k+1}  = x_k   + v_k \cos\psi_k \cdot \Delta t$" "\n"
        r"$y_{k+1}  = y_k   + v_k \sin\psi_k \cdot \Delta t$" "\n"
        r"$\psi_{k+1} = \psi_k + v_k \kappa_k \cdot \Delta t$"
    )
    ax_traj.text(
        0.98, 0.04, eq_text,
        transform=ax_traj.transAxes,
        fontsize=9.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="#37474F", ec="none", alpha=0.88),
        color="white",
    )

    # Arrow indices label
    ax_traj.annotate(
        "Purple arrows\nshow heading (yaw)\nat waypoints 0,8,…,56",
        xy=(x[8], y[8]), xytext=(x[8] - 1.5, y[8] - 1.0),
        fontsize=8.5, color=C_ARROW,
        arrowprops=dict(arrowstyle="->", color=C_ARROW, lw=0.9),
    )

    # ── Save ───────────────────────────────────────────────────────────────
    plt.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {save_path}")


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pathlib

    OUT = pathlib.Path("/home/seungwoo/workspace/analysis/figures/action_trajectory.png")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    data = make_synthetic_actions(n=64, dt=0.1, seed=42)
    plot_action_trajectory(data, str(OUT))
