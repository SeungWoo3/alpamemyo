"""Demand Layering 결과 시각화"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RESULTS_DIR, "results.json")) as f:
    dl_results = json.load(f)

dl = dl_results[0]

# Figure 1: 방법별 추론 시간 & Peak VRAM 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

methods = ["FP16\nUnified Memory", "Demand\nLayering (FP16)", "4-bit\n(no swap)"]
infer_times = [273.79, dl["inference_time"], 4.79]
peak_vrams = [21.52, dl["peak_vram_gb"], 8.87]
colors = ["#e74c3c", "#2ecc71", "#3498db"]

# Panel 1: Inference Time
ax1 = axes[0]
bars = ax1.bar(methods, infer_times, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("Inference Time (seconds)", fontsize=12)
ax1.set_title("Inference Time Comparison", fontsize=13, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars, infer_times):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 3,
             f"{val:.1f}s", ha="center", va="bottom", fontsize=12, fontweight="bold")
# Speedup annotation
ax1.annotate(f"2.35x faster", xy=(1, dl["inference_time"]),
             xytext=(1.5, 200), fontsize=11, color="#27ae60", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5))

# Panel 2: Peak VRAM
ax2 = axes[1]
bars2 = ax2.bar(methods, peak_vrams, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
ax2.axhline(y=12, color="red", linestyle="--", linewidth=2, label="Physical VRAM (12GB)")
ax2.set_ylabel("Peak VRAM (GB)", fontsize=12)
ax2.set_title("Peak VRAM Usage", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars2, peak_vrams):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f"{val:.1f} GB", ha="center", va="bottom", fontsize=12, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "demand_layering_comparison.png"), dpi=150, bbox_inches="tight")
print("Saved: demand_layering_comparison.png")

# Figure 2: Transfer overhead breakdown
fig2, ax = plt.subplots(figsize=(10, 6))

vlm_stats = dl["vlm_transfer_stats"]
total_infer = dl["inference_time"]
h2d_time = vlm_stats["h2d_total_s"]
d2h_time = vlm_stats["d2h_total_s"]
compute_time = total_infer - h2d_time - d2h_time

categories = ["CPU→GPU\n(H2D)", "GPU→CPU\n(D2H)", "Compute\n(forward)"]
times = [h2d_time, d2h_time, compute_time]
colors2 = ["#e67e22", "#e74c3c", "#2ecc71"]

bars = ax.bar(categories, times, color=colors2, width=0.6, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Demand Layering: Inference Time Breakdown", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")

for bar, val in zip(bars, times):
    pct = val / total_infer * 100
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
            f"{val:.1f}s\n({pct:.0f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Summary text
ax.text(0.98, 0.95,
        f"Total: {total_infer:.1f}s\n"
        f"Transfers: {vlm_stats['total_transfers']} ops\n"
        f"H2D avg: {vlm_stats['h2d_avg_ms']:.1f}ms\n"
        f"D2H avg: {vlm_stats['d2h_avg_ms']:.1f}ms",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig2.savefig(os.path.join(RESULTS_DIR, "transfer_breakdown.png"), dpi=150, bbox_inches="tight")
print("Saved: transfer_breakdown.png")

# Figure 3: Architecture diagram (text)
fig3, ax = plt.subplots(figsize=(14, 7))
ax.axis("off")
ax.set_title("Demand Layering Architecture for Alpamayo-R1-10B", fontsize=14, fontweight="bold", pad=20)

arch_text = """
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU (12GB Physical VRAM)                         │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Vision Encoder    │  │ VLM Non-Layer│  │ Expert (36 layers)  │  │
│  │ (1.15 GB)         │  │ (embed, norm)│  │ (4.56 GB) - always  │  │
│  └──────────────────┘  │ (0.85 GB)    │  │ GPU-resident        │  │
│                         └──────────────┘  └─────────────────────┘  │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐    │
│  │ Action Projections│  │ VLM Layers 30-35 (6 layers, GPU)    │    │
│  │ (0.30 GB)         │  │ (2.52 GB) - always GPU-resident     │    │
│  └──────────────────┘  └──────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────┐                    │
│  │ Active VLM Layer (on-demand, ~0.42 GB)     │ ← Hook loads      │
│  │ One layer at a time during VLM forward     │   from CPU         │
│  └────────────────────────────────────────────┘                    │
│                         Peak VRAM: 11.03 GB                        │
├─────────────────────────────────────────────────────────────────────┤
│                    CPU (16GB System RAM)                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ VLM Layers 0-29 (30 layers, ~12.6 GB)                     │    │
│  │ Stored on CPU, loaded to GPU one-at-a-time via hooks      │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
"""
ax.text(0.5, 0.5, arch_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig3.savefig(os.path.join(RESULTS_DIR, "architecture_diagram.png"), dpi=150, bbox_inches="tight")
print("Saved: architecture_diagram.png")

print("\nAll visualizations generated!")
