#!/usr/bin/env python3
"""Part 5: 시각화

CPU-GPU 메모리 스와핑 비효율 분석 결과를 시각화한다.

1. Unified Memory 스와핑 패턴 시계열
2. 전송 크기별 대역폭 비교 (pinned vs pageable)
3. 스왑 전략 비교 차트
4. WSL2 오버헤드 분석
5. 전송 granularity vs 대역폭
"""

import os
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ─── Config ─────────────────────────────────────────────────────
OUTPUT_DIR = "/home/seungwoo/workspace/research/03-swap-optimization"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
FONT_PATH = "/home/seungwoo/.local/share/fonts/NanumGothic-Regular.ttf"
FONT_BOLD_PATH = "/home/seungwoo/.local/share/fonts/NanumGothic-Bold.ttf"

os.makedirs(FIG_DIR, exist_ok=True)

# Korean font
for fpath in [FONT_PATH, FONT_BOLD_PATH]:
    if os.path.exists(fpath):
        fm.fontManager.addfont(fpath)
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.15


# ─── Load Data ──────────────────────────────────────────────────

def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path) as f:
        return json.load(f)

def load_csv(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    return data


# ============================================================
# Figure 1: Unified Memory Timeline
# ============================================================

def figure_unified_memory_timeline():
    """고빈도 메모리 모니터링 시계열."""
    print("Creating figure 1: Unified Memory Timeline...")

    csv_data = load_csv("unified_memory_timeline.csv")
    if not csv_data:
        print("  No timeline data available")
        return

    times = [d["time_ms"] / 1000 for d in csv_data]  # Convert to seconds
    allocated = [d["allocated_mb"] / 1024 for d in csv_data]  # Convert to GB
    reserved = [d["reserved_mb"] / 1024 for d in csv_data]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(times, 0, reserved, alpha=0.2, color="C0", label="예약됨 (Reserved)")
    ax.fill_between(times, 0, allocated, alpha=0.4, color="C1", label="할당됨 (Allocated)")
    ax.plot(times, allocated, color="C1", linewidth=1.0)
    ax.plot(times, reserved, color="C0", linewidth=1.0, linestyle="--")

    # Physical VRAM limit
    ax.axhline(y=12.0, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="물리 VRAM 한계 (12 GB)")

    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("메모리 (GB)")
    ax.set_title("Unified Memory 모니터링 시계열\n(10ms 간격 샘플링)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(max(reserved) * 1.1, 13))
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(FIG_DIR, "01_unified_memory_timeline.png"))
    plt.close(fig)
    print("  Saved 01_unified_memory_timeline.png")


# ============================================================
# Figure 2: Bandwidth Comparison (Pinned vs Pageable)
# ============================================================

def figure_bandwidth_comparison():
    """Pinned vs Pageable 대역폭 비교."""
    print("Creating figure 2: Bandwidth Comparison...")

    data = load_json("pinned_transfer_results.json")
    bw_data = data.get("exp1_bandwidth", [])

    if not bw_data:
        print("  No bandwidth data available")
        return

    sizes = [d["size_mb"] for d in bw_data if "size_mb" in d]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # H2D
    ax = axes[0]
    pinned_h2d = [d.get("pinned_h2d_bw_gbs", 0) for d in bw_data]
    pageable_h2d = [d.get("pageable_h2d_bw_gbs", 0) for d in bw_data]

    x = np.arange(len(sizes))
    width = 0.35
    bars1 = ax.bar(x - width/2, pinned_h2d, width, label="Pinned", color="C0", alpha=0.8)
    bars2 = ax.bar(x + width/2, pageable_h2d, width, label="Pageable", color="C3", alpha=0.8)

    ax.axhline(y=15.75, color="gray", linestyle=":", linewidth=1, label="PCIe 3.0 x16 이론 (15.75 GB/s)")
    ax.set_xlabel("전송 크기 (MB)")
    ax.set_ylabel("대역폭 (GB/s)")
    ax.set_title("CPU -> GPU (H2D) 대역폭")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    # D2H
    ax = axes[1]
    pinned_d2h = [d.get("pinned_d2h_bw_gbs", 0) for d in bw_data]
    pageable_d2h = [d.get("pageable_d2h_bw_gbs", 0) for d in bw_data]

    bars1 = ax.bar(x - width/2, pinned_d2h, width, label="Pinned", color="C0", alpha=0.8)
    bars2 = ax.bar(x + width/2, pageable_d2h, width, label="Pageable", color="C3", alpha=0.8)

    ax.axhline(y=15.75, color="gray", linestyle=":", linewidth=1, label="PCIe 3.0 x16 이론")
    ax.set_xlabel("전송 크기 (MB)")
    ax.set_ylabel("대역폭 (GB/s)")
    ax.set_title("GPU -> CPU (D2H) 대역폭")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Pinned vs Pageable 메모리 전송 대역폭 비교 (WSL2)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_bandwidth_comparison.png"))
    plt.close(fig)
    print("  Saved 02_bandwidth_comparison.png")


# ============================================================
# Figure 3: Transfer Granularity vs Bandwidth
# ============================================================

def figure_granularity():
    """전송 granularity에 따른 대역폭 변화."""
    print("Creating figure 3: Transfer Granularity...")

    data = load_json("prefetch_results.json")
    gran_data = data.get("exp3_granularity", [])

    if not gran_data:
        print("  No granularity data available")
        return

    chunks = [d["chunk_mb"] for d in gran_data if "chunk_mb" in d and "h2d_bw_gbs" in d]
    bw = [d["h2d_bw_gbs"] for d in gran_data if "chunk_mb" in d and "h2d_bw_gbs" in d]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(chunks, bw, "o-", color="C0", linewidth=2, markersize=8, label="실측 대역폭")

    # Highlight optimal range
    max_bw = max(bw)
    optimal_idx = bw.index(max_bw)
    ax.axhspan(max_bw * 0.9, max_bw * 1.1, alpha=0.15, color="green", label=f"최적 범위 (>{max_bw*0.9:.1f} GB/s)")
    ax.scatter([chunks[optimal_idx]], [max_bw], color="red", s=150, zorder=5, label=f"최적점: {chunks[optimal_idx]}MB ({max_bw:.1f} GB/s)")

    ax.axhline(y=15.75, color="gray", linestyle=":", linewidth=1, label="PCIe 3.0 x16 이론 (15.75 GB/s)")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("청크 크기 (MB)")
    ax.set_ylabel("유효 대역폭 (GB/s)")
    ax.set_title("전송 단위(Granularity) vs 유효 대역폭\n(512MB 총량을 N개 청크로 분할 전송)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(f"단일 전송\n{chunks[-1]}MB\n{bw[-1]:.1f} GB/s",
                xy=(chunks[-1], bw[-1]), xytext=(chunks[-1]*0.5, bw[-1]-2),
                arrowprops=dict(arrowstyle="->", color="C3"),
                fontsize=8, color="C3")
    ax.annotate(f"작은 청크\n{chunks[0]}MB\n{bw[0]:.1f} GB/s",
                xy=(chunks[0], bw[0]), xytext=(chunks[0]*4, bw[0]-2),
                arrowprops=dict(arrowstyle="->", color="C3"),
                fontsize=8, color="C3")

    fig.savefig(os.path.join(FIG_DIR, "03_granularity_bandwidth.png"))
    plt.close(fig)
    print("  Saved 03_granularity_bandwidth.png")


# ============================================================
# Figure 4: WSL2 Overhead Analysis
# ============================================================

def figure_wsl2_overhead():
    """WSL2 오버헤드 종합 분석."""
    print("Creating figure 4: WSL2 Overhead...")

    data = load_json("wsl2_overhead_results.json")
    wsl2 = data.get("exp5_wsl2", {})
    bidir = data.get("exp4_bidirectional", {})
    latency = data.get("exp3_latency", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) PCIe Bandwidth Efficiency
    ax = axes[0][0]
    pcie = wsl2.get("pcie_analysis", {})
    if pcie and "error" not in pcie:
        categories = ["H2D (CPU->GPU)", "D2H (GPU->CPU)"]
        theoretical = [15.75, 15.75]
        actual = [pcie.get("pinned_h2d_bw_gbs", 0), pcie.get("pinned_d2h_bw_gbs", 0)]
        overhead = [15.75 - a for a in actual]

        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x, actual, width, label="실측 대역폭", color="C0", alpha=0.8)
        ax.bar(x, overhead, width, bottom=actual, label="WSL2 오버헤드", color="C3", alpha=0.5)

        for i, (a, t) in enumerate(zip(actual, theoretical)):
            pct = a / t * 100
            ax.text(i, a/2, f"{a:.1f} GB/s\n({pct:.0f}%)", ha="center", va="center", fontsize=10, fontweight="bold")
            ax.text(i, a + (t-a)/2, f"{t-a:.1f} GB/s\n({100-pct:.0f}%)", ha="center", va="center", fontsize=9, color="C3")

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel("대역폭 (GB/s)")
        ax.set_title("PCIe 3.0 x16 대역폭 효율\n(Pinned Memory, WSL2)")
        ax.legend()
        ax.set_ylim(0, 18)

    # (0,1) Pageable D2H Anomaly
    ax = axes[0][1]
    anomaly = wsl2.get("pageable_d2h_anomaly", [])
    if anomaly:
        sizes = [d["size_mb"] for d in anomaly if "size_mb" in d and "pageable_d2h_bw" in d]
        pageable_bw = [d["pageable_d2h_bw"] for d in anomaly if "size_mb" in d and "pageable_d2h_bw" in d]
        pinned_bw = [d["pinned_d2h_bw"] for d in anomaly if "size_mb" in d and "pinned_d2h_bw" in d]

        x = np.arange(len(sizes))
        width = 0.35
        ax.bar(x - width/2, pageable_bw, width, label="Pageable D2H", color="C3", alpha=0.8)
        ax.bar(x + width/2, pinned_bw, width, label="Pinned D2H", color="C0", alpha=0.8)

        for i, (p, pi) in enumerate(zip(pageable_bw, pinned_bw)):
            slowdown = pi / max(p, 0.001)
            ax.text(i, max(p, pi) + 0.5, f"{slowdown:.1f}x", ha="center", fontsize=9, fontweight="bold", color="C3")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}MB" for s in sizes])
        ax.set_ylabel("대역폭 (GB/s)")
        ax.set_title("WSL2 Pageable D2H 비정상\n(Pageable이 Pinned 대비 ~4.5x 느림)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    # (1,0) Bidirectional Contention
    ax = axes[1][0]
    if bidir and "error" not in bidir:
        categories = ["H2D 단독", "D2H 단독", "양방향\n(총 시간)", "순차\n(H2D+D2H)"]
        times = [bidir["h2d_only_ms"], bidir["d2h_only_ms"],
                 bidir["bidir_total_ms"], bidir["sequential_ms"]]
        colors = ["C0", "C1", "C2", "C3"]

        bars = ax.bar(categories, times, color=colors, alpha=0.8)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{t:.1f}ms", ha="center", fontsize=9)

        overlap = bidir.get("overlap_ratio", 0)
        ax.text(0.5, 0.95, f"중첩 비율: {overlap:.2f}x", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        ax.set_ylabel("전송 시간 (ms)")
        ax.set_title(f"양방향 전송 경합 분석\n({bidir.get('size_mb', '?')}MB)")
        ax.grid(True, axis="y", alpha=0.3)

    # (1,1) Transfer Latency
    ax = axes[1][1]
    if latency:
        valid = [d for d in latency if "size_bytes" in d and "h2d_avg_ms" in d]
        if valid:
            sizes_bytes = [d["size_bytes"] for d in valid]
            avg_ms = [d["h2d_avg_ms"] for d in valid]
            min_ms = [d["h2d_min_ms"] for d in valid]

            ax.plot(sizes_bytes, avg_ms, "o-", label="평균", color="C0", markersize=6)
            ax.plot(sizes_bytes, min_ms, "s--", label="최소", color="C1", markersize=5)

            # Fixed overhead line
            fixed = valid[0]["h2d_min_ms"]
            ax.axhline(y=fixed, color="red", linestyle=":", alpha=0.5,
                       label=f"고정 오버헤드: {fixed:.3f}ms")

            ax.set_xscale("log")
            ax.set_xlabel("전송 크기 (bytes)")
            ax.set_ylabel("전송 시간 (ms)")
            ax.set_title("전송 크기 vs 지연 시간\n(고정 오버헤드 + 크기 비례)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle("WSL2 메모리 전송 오버헤드 종합 분석", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_wsl2_overhead.png"))
    plt.close(fig)
    print("  Saved 04_wsl2_overhead.png")


# ============================================================
# Figure 5: Swap Strategy Comparison
# ============================================================

def figure_strategy_comparison():
    """스왑 전략 비교 종합 차트."""
    print("Creating figure 5: Strategy Comparison...")

    # Collect key metrics from all experiments
    pinned_data = load_json("pinned_transfer_results.json")
    prefetch_data = load_json("prefetch_results.json")
    um_data = load_json("unified_memory_results.json")
    wsl2_data = load_json("wsl2_overhead_results.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Memory Overcommit Behavior
    ax = axes[0][0]
    um_pressure = um_data.get("exp1_memory_stats", {}).get("sequential_pressure", [])
    if um_pressure:
        chunks = [d["chunk_idx"] for d in um_pressure if "chunk_idx" in d]
        alloc_times = [d["alloc_time_s"] * 1000 for d in um_pressure if "alloc_time_s" in d]
        total_gb = [d.get("total_allocated_gb", 0) for d in um_pressure if "total_allocated_gb" in d]

        ax2 = ax.twinx()
        line1 = ax.bar(chunks[:len(alloc_times)], alloc_times, color="C0", alpha=0.7, label="할당 시간")
        line2 = ax2.plot(chunks[:len(total_gb)], total_gb, "o-", color="C3", label="누적 VRAM")
        ax2.axhline(y=12.0, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="물리 VRAM (12GB)")

        ax.set_xlabel("청크 인덱스 (1GB/청크)")
        ax.set_ylabel("할당 시간 (ms)")
        ax2.set_ylabel("누적 할당 (GB)")
        ax.set_title("CUDA 메모리 오버커밋 동작\n(12GB 초과 시 할당 시간 급증)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # (0,1) Prefetch Overlap Effect
    ax = axes[0][1]
    overlap_data = prefetch_data.get("exp2_prefetch_overlap", [])
    if overlap_data:
        sizes = [d["transfer_mb"] for d in overlap_data if "transfer_mb" in d and "sequential_ms" in d]
        seq = [d["sequential_ms"] for d in overlap_data if "transfer_mb" in d and "sequential_ms" in d]
        ovlp = [d["overlapped_ms"] for d in overlap_data if "transfer_mb" in d and "overlapped_ms" in d]
        savings = [d.get("savings_pct", 0) for d in overlap_data if "transfer_mb" in d and "savings_pct" in d]

        x = np.arange(len(sizes))
        width = 0.35
        ax.bar(x - width/2, seq, width, label="순차 (전송 + 연산)", color="C3", alpha=0.8)
        ax.bar(x + width/2, ovlp, width, label="중첩 (스트림 기반)", color="C0", alpha=0.8)

        for i, s in enumerate(savings):
            ax.text(i, max(seq[i], ovlp[i]) + 2, f"-{s:.0f}%", ha="center", fontsize=9, color="C0", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}MB" for s in sizes])
        ax.set_xlabel("전송 크기")
        ax.set_ylabel("시간 (ms)")
        ax.set_title("CUDA 스트림 전송-연산 중첩 효과")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    # (1,0) Layer-wise Offloading Comparison
    ax = axes[1][0]
    layerwise = pinned_data.get("exp3_layerwise", {})
    comparison = layerwise.get("comparison", {})
    if comparison:
        strategies = ["동기식\n순차 오프로딩", "비동기\n파이프라인"]
        times = [comparison["sync_total_ms"], comparison["async_total_ms"]]
        colors = ["C3", "C0"]

        bars = ax.bar(strategies, times, color=colors, alpha=0.8, width=0.5)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f"{t:.0f}ms", ha="center", fontsize=11, fontweight="bold")

        speedup = comparison.get("speedup", 0)
        ax.text(0.5, 0.9, f"스피드업: {speedup:.2f}x", transform=ax.transAxes,
                ha="center", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        n = comparison.get("n_layers", 0)
        sz = comparison.get("layer_size_mb", 0)
        ax.set_ylabel("총 시간 (ms)")
        ax.set_title(f"레이어별 오프로딩 전략 비교\n({n} layers x {sz:.0f}MB)")
        ax.grid(True, axis="y", alpha=0.3)

    # (1,1) Overcommit Access Time
    ax = axes[1][1]
    overcommit = prefetch_data.get("exp5_unified_wsl2", {}).get("overcommit", [])
    pressure = prefetch_data.get("exp5_unified_wsl2", {}).get("memory_pressure", [])
    if overcommit:
        valid = [d for d in overcommit if d.get("status") == "success"]
        gb = [d["target_gb"] for d in valid]
        alloc_s = [d["alloc_time_s"] for d in valid]
        access_ms = [d["access_time_ms"] for d in valid]

        ax2 = ax.twinx()
        line1 = ax.bar(gb, alloc_s, width=1.5, color="C0", alpha=0.6, label="할당 시간 (초)")
        line2 = ax2.plot(gb, access_ms, "ro-", label="접근 시간 (ms)", markersize=8, linewidth=2)
        ax.axvline(x=12, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="물리 VRAM (12GB)")

        ax.set_xlabel("할당 크기 (GB)")
        ax.set_ylabel("할당 시간 (초)")
        ax2.set_ylabel("접근 시간 (ms)")
        ax.set_title("오버커밋 할당/접근 시간\n(12GB 초과 시 접근 시간 급증)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    fig.suptitle("CPU-GPU 스왑 전략 비교 종합", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_strategy_comparison.png"))
    plt.close(fig)
    print("  Saved 05_strategy_comparison.png")


# ============================================================
# Figure 6: Transfer Size Scaling (Detailed)
# ============================================================

def figure_transfer_scaling():
    """전송 크기별 대역폭 스케일링 상세."""
    print("Creating figure 6: Transfer Size Scaling...")

    data = load_json("pinned_transfer_results.json")
    scaling = data.get("exp4_size_scaling", [])

    if not scaling:
        print("  No scaling data available")
        return

    valid = [d for d in scaling if "size_mb" in d and "h2d_bw_gbs" in d]
    if not valid:
        return

    sizes = [d["size_mb"] for d in valid]
    h2d_bw = [d["h2d_bw_gbs"] for d in valid]
    d2h_bw = [d["d2h_bw_gbs"] for d in valid]
    asymmetry = [d.get("asymmetry_ratio", 0) for d in valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bandwidth vs Size
    ax1.semilogx(sizes, h2d_bw, "o-", label="H2D (CPU->GPU)", color="C0", markersize=8, linewidth=2)
    ax1.semilogx(sizes, d2h_bw, "s-", label="D2H (GPU->CPU)", color="C1", markersize=8, linewidth=2)
    ax1.axhline(y=15.75, color="gray", linestyle=":", linewidth=1, label="PCIe 3.0 이론")

    # Add Alpamayo layer size markers
    alpamayo_sizes = [
        (386, "VLM Layer"),
        (1150, "Vision Enc"),
    ]
    for asize, alabel in alpamayo_sizes:
        if min(sizes) <= asize <= max(sizes):
            ax1.axvline(x=asize, color="green", linestyle="--", alpha=0.5)
            ax1.text(asize * 1.05, 1, alabel, fontsize=8, color="green", rotation=90, va="bottom")

    ax1.set_xlabel("전송 크기 (MB)")
    ax1.set_ylabel("대역폭 (GB/s)")
    ax1.set_title("Pinned Memory 전송 크기별 대역폭\n(CUDA Event 기반 정밀 측정)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 16)

    # Asymmetry Ratio
    ax2.semilogx(sizes, asymmetry, "D-", color="C3", markersize=8, linewidth=2)
    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, label="대칭 (1.0)")
    ax2.fill_between(sizes, 1.0, asymmetry, alpha=0.2, color="C3")

    ax2.set_xlabel("전송 크기 (MB)")
    ax2.set_ylabel("H2D/D2H 비율")
    ax2.set_title("H2D vs D2H 비대칭 비율\n(<1: D2H가 더 빠름, >1: H2D가 더 빠름)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add explanation
    avg_asym = np.mean(asymmetry)
    direction = "D2H (GPU->CPU) 우위" if avg_asym < 1 else "H2D (CPU->GPU) 우위"
    ax2.text(0.5, 0.1, f"평균 비율: {avg_asym:.2f}\n({direction})",
             transform=ax2.transAxes, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("전송 크기별 Pinned Memory 대역폭 스케일링", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_transfer_scaling.png"))
    plt.close(fig)
    print("  Saved 06_transfer_scaling.png")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Part 5: 시각화 생성")
    print("=" * 70)

    figure_unified_memory_timeline()
    figure_bandwidth_comparison()
    figure_granularity()
    figure_wsl2_overhead()
    figure_strategy_comparison()
    figure_transfer_scaling()

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
