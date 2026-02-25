"""4-bit VRAM 제한별 성능 비교 시각화"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(RESULTS_DIR, "results.json")) as f:
    results = json.load(f)

vram_limits = [r["vram_limit_gb"] for r in results]
inference_times = [r["inference_time"] for r in results]
load_times = [r["model_load_time"] for r in results]
peak_vrams = [r["peak_vram_gb"] for r in results]

labels = [f"{v}GB" for v in vram_limits]
x = np.arange(len(labels))

# Figure 1: 추론 시간 비교 (로그 스케일)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Inference Time (log scale)
ax1 = axes[0]
bars = ax1.bar(x, inference_times, color=["#2ecc71", "#3498db", "#e67e22", "#e74c3c"], width=0.6, edgecolor="black", linewidth=0.5)
ax1.set_yscale("log")
ax1.set_xlabel("Available VRAM", fontsize=12)
ax1.set_ylabel("Inference Time (seconds, log scale)", fontsize=12)
ax1.set_title("4-bit Model Inference Time vs VRAM Limit", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars, inference_times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
             f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
# Add slowdown annotations
baseline = inference_times[0]
for i, (bar, val) in enumerate(zip(bars, inference_times)):
    if i > 0:
        slowdown = val / baseline
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 0.4,
                 f"{slowdown:.0f}x", ha="center", va="center", fontsize=10,
                 color="white", fontweight="bold")

# Panel 2: Peak VRAM & Physical VRAM line
ax2 = axes[1]
bars2 = ax2.bar(x, peak_vrams, color=["#2ecc71", "#3498db", "#e67e22", "#e74c3c"], width=0.6, edgecolor="black", linewidth=0.5)
ax2.axhline(y=12, color="red", linestyle="--", linewidth=2, label="Physical VRAM (12GB)")
ax2.axhline(y=8.87, color="green", linestyle=":", linewidth=1.5, label="Model Size (8.87GB)")
ax2.set_xlabel("Available VRAM", fontsize=12)
ax2.set_ylabel("Peak VRAM Usage (GB)", fontsize=12)
ax2.set_title("Peak VRAM Usage (incl. Unified Memory)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars2, peak_vrams):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2,
             f"{val:.1f}GB", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Panel 3: Load Time comparison
ax3 = axes[2]
bars3 = ax3.bar(x, load_times, color=["#2ecc71", "#3498db", "#e67e22", "#e74c3c"], width=0.6, edgecolor="black", linewidth=0.5)
ax3.set_xlabel("Available VRAM", fontsize=12)
ax3.set_ylabel("Model Load Time (seconds)", fontsize=12)
ax3.set_title("Model Load Time vs VRAM Limit", fontsize=13, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars3, load_times):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 2,
             f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "vram_limit_comparison.png"), dpi=150, bbox_inches="tight")
print(f"Saved: vram_limit_comparison.png")

# Figure 2: 성능 저하 곡선 (Inference Time vs Available VRAM)
fig2, ax = plt.subplots(figsize=(10, 6))
available_vrams = [r["available_vram_gb"] for r in results]

ax.plot(available_vrams, inference_times, "o-", color="#e74c3c", linewidth=2.5, markersize=10, markerfacecolor="white", markeredgewidth=2)
for av, it in zip(available_vrams, inference_times):
    ax.annotate(f"{it:.1f}s", (av, it), textcoords="offset points",
                xytext=(10, 10), fontsize=11, fontweight="bold")

# Cliff region shading
ax.axvspan(8.5, 10.5, alpha=0.15, color="orange", label="Cliff region (~8.87GB model size)")
ax.axvline(x=8.87, color="green", linestyle="--", linewidth=1.5, label="4-bit Model Size (8.87GB)")

ax.set_xlabel("Available VRAM (GB)", fontsize=13)
ax.set_ylabel("Inference Time (seconds)", fontsize=13)
ax.set_title("Performance Cliff: 4-bit Model Under VRAM Constraints", fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
ax.invert_xaxis()  # Higher VRAM on left

fig2.savefig(os.path.join(RESULTS_DIR, "performance_cliff.png"), dpi=150, bbox_inches="tight")
print(f"Saved: performance_cliff.png")

# Figure 3: Summary table as image
fig3, ax = plt.subplots(figsize=(12, 4))
ax.axis("off")
ax.set_title("4-bit Alpamayo VRAM Limit Comparison — Summary", fontsize=14, fontweight="bold", pad=20)

table_data = [
    ["VRAM Limit", "Available", "Load Time", "Inference Time", "Peak VRAM", "Slowdown"],
]
for r in results:
    slowdown = f"{r['inference_time'] / results[0]['inference_time']:.1f}x"
    table_data.append([
        f"{r['vram_limit_gb']} GB",
        f"{r['available_vram_gb']:.1f} GB",
        f"{r['model_load_time']:.1f}s",
        f"{r['inference_time']:.1f}s",
        f"{r['peak_vram_gb']:.1f} GB",
        slowdown if r != results[0] else "baseline",
    ])

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color coding
colors = ["#d5f5e3", "#d6eaf8", "#fdebd0", "#fadbd8"]
for i in range(4):
    for j in range(6):
        table[i+1, j].set_facecolor(colors[i])
for j in range(6):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

fig3.savefig(os.path.join(RESULTS_DIR, "summary_table.png"), dpi=150, bbox_inches="tight")
print(f"Saved: summary_table.png")

print("\nAll visualizations generated!")
