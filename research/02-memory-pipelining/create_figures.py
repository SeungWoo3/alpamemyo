#!/usr/bin/env python3
"""
Research Direction 2: Memory Pipelining Visualization

Figures:
1. Inference pipeline diagram (Alpamayo 3-phase pipeline)
2. Sequential offloading VRAM timeline
3. Pipeline strategy comparison (baseline vs sequential vs async)
4. PCIe bandwidth measurements
5. Layerwise offloading analysis
"""

import json
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "/home/seungwoo/workspace/research/02-memory-pipelining/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = "/home/seungwoo/workspace/research/02-memory-pipelining"


def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return list(csv.DictReader(f))


# ============================================================
# Figure 1: Inference Pipeline Diagram
# ============================================================

def create_pipeline_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1, 11)
    ax.axis("off")
    ax.set_title("Alpamayo-R1 Inference Pipeline & Sequential Module Offloading",
                 fontsize=15, fontweight="bold", pad=20)

    colors = {
        "vision": "#4CAF50", "vlm": "#2196F3", "expert": "#FF9800",
        "kv_cache": "#E91E63", "gpu": "#FFD54F", "arrow": "#424242",
    }

    # Phase boxes
    for (x, w, name, sub, color) in [
        (0.5, 4, "Phase 1: Vision Encoder", "576M params | 1.15 GB", colors["vision"]),
        (5.5, 4.5, "Phase 2: VLM (Qwen3-VL-8B)", "8.2B params | 16.44 GB | 36 layers", colors["vlm"]),
        (10.8, 4.2, "Phase 3: Expert+Diffusion", "2.3B params | 4.56 GB | 10 steps", colors["expert"]),
    ]:
        box = FancyBboxPatch((x, 8.5), w, 1.5, boxstyle="round,pad=0.1",
                              facecolor=color, alpha=0.8, edgecolor="black", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, 9.5, name, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(x + w / 2, 8.9, sub, ha="center", va="center", fontsize=8, color="white")

    # Arrows between phases
    ax.annotate("", xy=(5.5, 9.25), xytext=(4.5, 9.25),
                arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2))
    ax.annotate("", xy=(10.8, 9.25), xytext=(10.0, 9.25),
                arrowprops=dict(arrowstyle="->", color=colors["arrow"], lw=2))
    ax.text(5.0, 9.65, "Embeddings\n~few MB", ha="center", va="bottom", fontsize=7,
            color=colors["arrow"], style="italic")
    ax.text(10.4, 9.65, "KV Cache\n+ tokens", ha="center", va="bottom", fontsize=7,
            color=colors["arrow"], style="italic")

    # Sequential Offloading scenario
    y_base = 5.5
    gpu_box = FancyBboxPatch((0.3, y_base), 15, 2.2, boxstyle="round,pad=0.1",
                              facecolor=colors["gpu"], alpha=0.3, edgecolor="#FFC107", linewidth=2, linestyle="--")
    ax.add_patch(gpu_box)
    ax.text(0.8, y_base + 2.0, "GPU VRAM (12 GB)", fontsize=10, fontweight="bold", color="#F57F17")

    # Time axis
    for i, tl in enumerate(["t=0", "t=1", "t=2", "t=3", "t=4", "t=5"]):
        ax.text(1 + i * 2.5, y_base - 0.3, tl, ha="center", fontsize=8, color="gray")

    # Vision Encoder on GPU
    ve = FancyBboxPatch((1, y_base + 0.3), 2, 0.6, boxstyle="round,pad=0.05",
                         facecolor=colors["vision"], alpha=0.9, edgecolor="black")
    ax.add_patch(ve)
    ax.text(2, y_base + 0.6, "Vision\n1.15GB", ha="center", va="center", fontsize=7, color="white")

    # VLM on GPU (exceeds VRAM)
    vlm_box = FancyBboxPatch((3.5, y_base + 0.3), 5, 1.8, boxstyle="round,pad=0.05",
                              facecolor=colors["vlm"], alpha=0.3, edgecolor=colors["vlm"], linewidth=2, linestyle="--")
    ax.add_patch(vlm_box)
    ax.text(6, y_base + 1.5, "VLM 16.44GB > 12GB VRAM!", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#D32F2F")
    ax.text(6, y_base + 0.9, "Layer-wise offloading required (~0.39GB each)", ha="center", va="center",
            fontsize=8, color=colors["vlm"])

    # VLM individual layers
    for i in range(3):
        lyr = FancyBboxPatch((3.8 + i * 1.5, y_base + 0.3), 1.2, 0.4, boxstyle="round,pad=0.03",
                              facecolor=colors["vlm"], alpha=0.8, edgecolor="black")
        ax.add_patch(lyr)
        ax.text(4.4 + i * 1.5, y_base + 0.5, f"L{i}", ha="center", va="center", fontsize=7, color="white")

    # Expert on GPU
    exp = FancyBboxPatch((9, y_base + 0.3), 3, 1.2, boxstyle="round,pad=0.05",
                          facecolor=colors["expert"], alpha=0.9, edgecolor="black")
    ax.add_patch(exp)
    ax.text(10.5, y_base + 0.9, "Expert\n4.56GB", ha="center", va="center", fontsize=8, color="white")

    # KV Cache dependency
    kv = FancyBboxPatch((9.3, y_base + 1.6), 2.4, 0.4, boxstyle="round,pad=0.03",
                         facecolor=colors["kv_cache"], alpha=0.7, edgecolor="black")
    ax.add_patch(kv)
    ax.text(10.5, y_base + 1.8, "KV Cache (~1-2GB)", ha="center", va="center", fontsize=7, color="white")

    # VRAM size comparison bars (bottom)
    y_bar = 1.5
    modules = [
        ("Vision Encoder", 1.15, colors["vision"]),
        ("VLM (BF16)", 16.44, colors["vlm"]),
        ("VLM Single Layer", 0.39, "#64B5F6"),
        ("Expert", 4.56, colors["expert"]),
        ("KV Cache (est.)", 1.5, colors["kv_cache"]),
    ]
    ax.text(0.5, y_bar + 1.2, "Module VRAM Size Comparison", fontsize=11, fontweight="bold")

    bar_max = 17
    for i, (name, size, color) in enumerate(modules):
        bar_width = min(size / bar_max * 12, 12)
        y = y_bar - i * 0.55
        bar = FancyBboxPatch((2.5, y - 0.15), bar_width, 0.35, boxstyle="round,pad=0.02",
                              facecolor=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.add_patch(bar)
        ax.text(2.3, y, name, ha="right", va="center", fontsize=8)
        ax.text(2.5 + bar_width + 0.2, y, f"{size:.2f} GB", ha="left", va="center", fontsize=8)

    limit_x = 2.5 + 12.0 / bar_max * 12
    ax.plot([limit_x, limit_x], [y_bar - len(modules) * 0.55, y_bar + 0.3],
            color="#F44336", linewidth=2, linestyle="--")
    ax.text(limit_x, y_bar + 0.4, "12GB Limit", ha="center", fontsize=8, color="#F44336", fontweight="bold")

    plt.savefig(os.path.join(OUTPUT_DIR, "01_pipeline_diagram.png"), facecolor="white")
    plt.close()
    print("Created: 01_pipeline_diagram.png")


# ============================================================
# Figure 2: VRAM Timeline
# ============================================================

def create_vram_timeline():
    csv_data = load_csv("sequential_offload_vram_timeline.csv")
    if not csv_data:
        print("No sequential offload timeline data")
        return

    times = [float(r["time_s"]) for r in csv_data]
    allocated = [float(r["allocated_gb"]) for r in csv_data]
    reserved = [float(r["reserved_gb"]) for r in csv_data]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(times, 0, reserved, alpha=0.2, color="#FF9800", label="Reserved")
    ax.fill_between(times, 0, allocated, alpha=0.5, color="#2196F3", label="Allocated")
    ax.plot(times, allocated, color="#1565C0", linewidth=1.5)
    ax.axhline(y=12.0, color="#F44336", linewidth=2, linestyle="--", label="12GB VRAM Limit")

    seq_results = load_json("sequential_offload_results.json")
    if seq_results:
        for p in seq_results.get("phases", []):
            if "vram_peak_gb" in p:
                peak = p["vram_peak_gb"]
                ax.axhline(y=peak, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
                ax.text(max(times) * 0.95, peak + 0.1, f"{p['phase']}: {peak:.2f}GB",
                        ha="right", fontsize=7, color="gray")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("VRAM Usage (GB)", fontsize=12)
    ax.set_title("VRAM Usage During Sequential Offloading Experiment", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(-0.2, 13)
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, "02_vram_timeline.png"), facecolor="white")
    plt.close()
    print("Created: 02_vram_timeline.png")


# ============================================================
# Figure 3: Strategy Comparison
# ============================================================

def create_strategy_comparison():
    async_data = load_json("async_transfer_results.json")
    seq_data = load_json("sequential_offload_results.json")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Chart 1: Peak VRAM comparison
    ax1 = axes[0]
    strategies = ["FP16\nBaseline\n(Full GPU)", "Sequential\nOffload\n(Module-level)", "Sequential\n+ Layer-wise\n(VLM)", "Sequential\nOffload\n(INT4 est.)"]
    vram_peaks = [21.52, 16.44, 4.57, 5.5]
    colors_bar = ["#F44336", "#FF9800", "#4CAF50", "#2196F3"]

    bars = ax1.bar(strategies, vram_peaks, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=12, color="#F44336", linewidth=2, linestyle="--", label="12GB Limit")
    ax1.set_ylabel("Peak VRAM (GB)", fontsize=11)
    ax1.set_title("Peak VRAM by Strategy", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)

    for bar, val in zip(bars, vram_peaks):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}GB", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars, vram_peaks):
        status = "FAIL" if val > 12 else "OK"
        color = "#F44336" if val > 12 else "#4CAF50"
        ax1.text(bar.get_x() + bar.get_width() / 2, 0.5,
                 status, ha="center", va="bottom", fontsize=10, fontweight="bold", color=color)
    ax1.set_ylim(0, 24)

    # Chart 2: Transfer overhead
    ax2 = axes[1]
    if async_data and "experiments" in async_data:
        te = next((e for e in async_data["experiments"] if e.get("method") == "transfer_estimates"), None)
        if te:
            labels = ["Synchronous\nOffloading", "Async\nPipeline"]
            values = [te["total_inference_est"]["sync_offload_overhead_s"],
                      te["total_inference_est"]["async_offload_overhead_s"]]
            bars2 = ax2.bar(labels, values, color=["#FF9800", "#4CAF50"], alpha=0.8,
                           edgecolor="black", linewidth=0.5, width=0.5)
            for bar, val in zip(bars2, values):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Transfer Overhead (s)", fontsize=11)
    ax2.set_title("Offloading Transfer Overhead", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Chart 3: Sync vs Async synthetic experiment
    ax3 = axes[2]
    if async_data and "comparison" in async_data:
        comp = async_data["comparison"]
        labels = ["Synchronous", "Async"]
        times_val = [comp["sync_time_s"], comp["async_time_s"]]
        vrams = [comp["sync_peak_vram_gb"], comp["async_peak_vram_gb"]]
        x = np.arange(len(labels))
        width = 0.3
        ax3.bar(x - width / 2, times_val, width, label="Total Time (s)", color="#2196F3", alpha=0.8, edgecolor="black")
        ax3.bar(x + width / 2, vrams, width, label="Peak VRAM (GB)", color="#FF9800", alpha=0.8, edgecolor="black")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend(fontsize=9)
        speedup = comp["speedup"]
        ax3.text(0.5, max(times_val) * 0.9,
                 f"Async Speedup: {speedup:.2f}x\nVRAM overhead: +{comp['vram_overhead_gb']*1000:.1f}MB",
                 ha="center", fontsize=9, style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax3.set_title("Synthetic Module Experiment\n(Sync vs Async)", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_strategy_comparison.png"), facecolor="white")
    plt.close()
    print("Created: 03_strategy_comparison.png")


# ============================================================
# Figure 4: PCIe Bandwidth
# ============================================================

def create_bandwidth_chart():
    async_data = load_json("async_transfer_results.json")
    if not async_data:
        return

    bw_exp = next((e for e in async_data["experiments"] if e.get("method") == "pcie_bandwidth"), None)
    if not bw_exp or "measurements" not in bw_exp:
        return

    measurements = bw_exp["measurements"]
    sizes = [m["size_mb"] for m in measurements]
    h2d_bw = [m["h2d_bandwidth_gbps"] for m in measurements]
    d2h_bw = [m["d2h_bandwidth_gbps"] for m in measurements]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sizes, h2d_bw, "o-", color="#2196F3", linewidth=2, markersize=6, label="CPU->GPU (H2D)")
    ax1.plot(sizes, d2h_bw, "s-", color="#FF9800", linewidth=2, markersize=6, label="GPU->CPU (D2H)")
    ax1.set_xlabel("Transfer Size (MB)", fontsize=11)
    ax1.set_ylabel("Bandwidth (GB/s)", fontsize=11)
    ax1.set_title("PCIe Bandwidth vs Transfer Size", fontsize=13, fontweight="bold")
    ax1.set_xscale("log")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    sustained_h2d = bw_exp.get("sustained_h2d_gbps", 0)
    sustained_d2h = bw_exp.get("sustained_d2h_gbps", 0)
    ax1.axhline(y=sustained_h2d, color="#1565C0", linewidth=1, linestyle="--", alpha=0.5)
    ax1.axhline(y=sustained_d2h, color="#E65100", linewidth=1, linestyle="--", alpha=0.5)
    ax1.text(max(sizes) * 0.8, sustained_h2d + 0.2, f"H2D: {sustained_h2d:.1f} GB/s", fontsize=8, color="#1565C0")
    ax1.text(max(sizes) * 0.8, sustained_d2h + 0.2, f"D2H: {sustained_d2h:.1f} GB/s", fontsize=8, color="#E65100")

    te = next((e for e in async_data["experiments"] if e.get("method") == "transfer_estimates"), None)
    if te:
        modules = te["per_module_estimates"]
        names = [m["module"] for m in modules if m["module"] != "VLM Full (IMPOSSIBLE)"]
        h2d_times = [m["h2d_time_s"] for m in modules if m["module"] != "VLM Full (IMPOSSIBLE)"]
        d2h_times = [m["d2h_time_s"] for m in modules if m["module"] != "VLM Full (IMPOSSIBLE)"]
        mod_sizes = [m["size_gb"] for m in modules if m["module"] != "VLM Full (IMPOSSIBLE)"]
        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width / 2, h2d_times, width, label="CPU->GPU", color="#2196F3", alpha=0.8)
        ax2.bar(x + width / 2, d2h_times, width, label="GPU->CPU", color="#FF9800", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("Transfer Time (s)", fontsize=11)
        ax2.set_title("Estimated Transfer Time per Module", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")
        for i, (_, size) in enumerate(zip(names, mod_sizes)):
            ax2.text(i, max(h2d_times[i], d2h_times[i]) + 0.02, f"{size:.2f}GB", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_pcie_bandwidth.png"), facecolor="white")
    plt.close()
    print("Created: 04_pcie_bandwidth.png")


# ============================================================
# Figure 5: Layerwise Analysis
# ============================================================

def create_layerwise_analysis():
    seq_data = load_json("sequential_offload_results.json")
    async_data = load_json("async_transfer_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Per-module transfer/inference times (measured)
    ax1 = axes[0, 0]
    if seq_data:
        phases = seq_data.get("phases", [])
        names = [p.get("phase", "?")[:22] for p in phases]
        h2d = [p.get("cpu_to_gpu_time_s", p.get("avg_cpu_to_gpu_s", 0)) for p in phases]
        d2h = [p.get("gpu_to_cpu_time_s", p.get("avg_gpu_to_cpu_s", 0)) for p in phases]
        inf = [p.get("inference_time_s", 0) for p in phases]
        x = np.arange(len(names))
        w = 0.25
        ax1.bar(x - w, h2d, w, label="CPU->GPU", color="#2196F3", alpha=0.8)
        ax1.bar(x, inf, w, label="Inference", color="#4CAF50", alpha=0.8)
        ax1.bar(x + w, d2h, w, label="GPU->CPU", color="#FF9800", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=8)
        ax1.set_ylabel("Time (s)", fontsize=10)
        ax1.set_title("Per-Module Transfer/Inference Time (Measured)", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis="y")

    # Chart 2: Per-module VRAM (measured)
    ax2 = axes[0, 1]
    if seq_data:
        phases = seq_data.get("phases", [])
        names = [p.get("phase", "?")[:22] for p in phases]
        vram_load = [p.get("vram_after_load_gb", 0) for p in phases]
        vram_peak = [p.get("vram_peak_gb", p.get("vram_peak_per_layer_gb", 0)) for p in phases]
        x = np.arange(len(names))
        w = 0.35
        ax2.bar(x - w / 2, vram_load, w, label="After Load", color="#2196F3", alpha=0.8)
        ax2.bar(x + w / 2, vram_peak, w, label="Peak", color="#F44336", alpha=0.8)
        ax2.axhline(y=12, color="#F44336", linewidth=2, linestyle="--", label="12GB Limit", alpha=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, fontsize=8)
        ax2.set_ylabel("VRAM (GB)", fontsize=10)
        ax2.set_title("Per-Module VRAM Usage (Measured)", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="y")
        for i, val in enumerate(vram_peak):
            ax2.text(i + w / 2, val + 0.1, f"{val:.2f}GB", ha="center", fontsize=7)

    # Chart 3: VLM layer transfer times (measured samples)
    ax3 = axes[1, 0]
    if seq_data:
        vlm = next((p for p in seq_data.get("phases", []) if p.get("phase") == "VLM Language Model"), None)
        if vlm and "layer_details" in vlm:
            details = vlm["layer_details"]
            ids = [d["layer_idx"] for d in details]
            h2d = [d["cpu_to_gpu_s"] for d in details]
            d2h = [d["gpu_to_cpu_s"] for d in details]
            x = np.arange(len(ids))
            w = 0.3
            ax3.bar(x - w / 2, h2d, w, label="CPU->GPU", color="#2196F3", alpha=0.8)
            ax3.bar(x + w / 2, d2h, w, label="GPU->CPU", color="#FF9800", alpha=0.8)
            ax3.set_xticks(x)
            ax3.set_xticklabels([f"Layer {i}" for i in ids])
            ax3.set_ylabel("Transfer Time (s)", fontsize=10)
            ax3.set_title("VLM Layer Transfer Time (Measured Samples)", fontsize=12, fontweight="bold")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis="y")
            avg_bw = vlm.get("estimated_bandwidth_gbps", 0)
            avg_size = vlm.get("avg_layer_size_gb", 0)
            ax3.text(0.5, max(h2d + d2h) * 0.95,
                     f"Layer size: {avg_size:.4f}GB\nEst. bandwidth: {avg_bw:.2f} GB/s\n"
                     f"36-layer total transfer: {vlm.get('estimated_total_transfer_s', 0):.1f}s",
                     fontsize=8, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Chart 4: Sync vs Async VLM layerwise estimate
    ax4 = axes[1, 1]
    if async_data:
        te = next((e for e in async_data["experiments"] if e.get("method") == "transfer_estimates"), None)
        if te and "vlm_layerwise" in te:
            vlm_est = te["vlm_layerwise"]
            labels = ["Synchronous", "Async Pipeline"]
            values = [vlm_est["sync_total_s"], vlm_est["pipeline_total_s"]]
            bars = ax4.bar(labels, values, color=["#FF9800", "#4CAF50"], alpha=0.8, edgecolor="black", width=0.5)
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                         f"{val:.1f}s", ha="center", fontsize=11, fontweight="bold")
            speedup = vlm_est["speedup"]
            ax4.text(0.5, max(values) * 0.7,
                     f"Speedup: {speedup:.1f}x",
                     ha="center", fontsize=14, fontweight="bold", color="#4CAF50",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax4.set_ylabel("Estimated Transfer Time (s)", fontsize=10)
            ax4.set_title("VLM 36-Layer Offloading Time Estimate", fontsize=12, fontweight="bold")
            ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_layerwise_analysis.png"), facecolor="white")
    plt.close()
    print("Created: 05_layerwise_analysis.png")


if __name__ == "__main__":
    print("Creating figures...")
    create_pipeline_diagram()
    create_vram_timeline()
    create_strategy_comparison()
    create_bandwidth_chart()
    create_layerwise_analysis()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
