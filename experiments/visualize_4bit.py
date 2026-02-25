#!/usr/bin/env python3
"""Alpamayo-R1-10B 4-bit 양자화 실험 결과 시각화 스크립트.

생성 차트:
  1. memory_timeline_4bit.png   - 4-bit VRAM 사용량 시계열
  2. phase_breakdown_4bit.png   - Phase별 소요 시간 분석
  3. comparison_fp16_vs_4bit.png - FP16 vs 4-bit 비교 차트
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────
# 한국어 폰트 설정
# ──────────────────────────────────────────────
from matplotlib import font_manager

font_path = "/home/seungwoo/.local/share/fonts/NanumGothic-Regular.ttf"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False
    FONT = "NanumGothic"
    print(f"[폰트] '{FONT}' 사용")
else:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    FONT = None
    print("[폰트] 한국어 폰트 없음 - DejaVu Sans fallback")

# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "memory_profile_4bit.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 150

PHASE_COLORS = {
    "init":              "#BDBDBD",
    "data_creation":     "#81D4FA",
    "model_load":        "#A5D6A7",
    "model_to_cuda":     "#FFCC80",
    "input_preparation": "#CE93D8",
    "vlm_inference":     "#EF9A9A",
}

PHASE_LABELS = {
    "init":              "Init",
    "data_creation":     "Data creation",
    "model_load":        "Model load (4-bit)",
    "model_to_cuda":     "Model .to(cuda)",
    "input_preparation": "Input preparation",
    "vlm_inference":     "VLM inference",
}

# ──────────────────────────────────────────────
# CSV 로드
# ──────────────────────────────────────────────
def load_csv(path):
    elapsed, allocated, reserved, phases = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed.append(float(row["elapsed_s"]))
            allocated.append(float(row["allocated_gb"]))
            reserved.append(float(row["reserved_gb"]))
            phases.append(row["phase"])
    return elapsed, allocated, reserved, phases


def get_phase_spans(elapsed, phases):
    """연속된 phase 구간의 (start, end, phase_name) 리스트를 반환."""
    spans = []
    cur_phase = phases[0]
    start = elapsed[0]
    for i in range(1, len(phases)):
        if phases[i] != cur_phase:
            spans.append((start, elapsed[i - 1], cur_phase))
            cur_phase = phases[i]
            start = elapsed[i]
    spans.append((start, elapsed[-1], cur_phase))
    return spans


# ──────────────────────────────────────────────
# Chart 1: VRAM 사용량 시계열 (4-bit)
# ──────────────────────────────────────────────
def chart_memory_timeline(elapsed, allocated, reserved, phases):
    fig, ax = plt.subplots(figsize=(14, 5))

    # phase 배경
    spans = get_phase_spans(elapsed, phases)
    for start, end, phase in spans:
        ax.axvspan(start, end, alpha=0.18, color=PHASE_COLORS.get(phase, "#EEEEEE"),
                   label=PHASE_LABELS.get(phase, phase))

    # 데이터 라인
    ax.plot(elapsed, allocated, linewidth=1.2, color="#1565C0", label="Allocated (GB)")
    ax.plot(elapsed, reserved, linewidth=1.2, color="#E65100", alpha=0.7, label="Reserved (GB)")

    # 물리 VRAM 한계선
    ax.axhline(y=12, color="red", linestyle="--", linewidth=1.5, label="Physical VRAM Limit 12GB")

    ax.set_xlabel("경과 시간 (초)" if FONT else "Elapsed Time (s)", fontsize=11)
    ax.set_ylabel("VRAM (GB)", fontsize=11)
    ax.set_title(
        "Alpamayo-R1-10B 4-bit VRAM 사용량 (RTX 3080 Ti 12GB)" if FONT
        else "Alpamayo-R1-10B 4-bit VRAM Usage (RTX 3080 Ti 12GB)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(elapsed[0], elapsed[-1])
    ax.set_ylim(0, max(max(reserved), 12) * 1.15)

    # 범례 - 중복 제거
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_h.append(h)
            unique_l.append(l)
    ax.legend(unique_h, unique_l, loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "memory_timeline_4bit.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────
# Chart 2: Phase별 소요 시간 수평 막대 (4-bit)
# ──────────────────────────────────────────────
def chart_phase_breakdown(elapsed, phases):
    # Phase별 시간 계산 (CSV 데이터에서 추출)
    spans = get_phase_spans(elapsed, phases)
    phase_times = {}
    for start, end, phase in spans:
        duration = end - start
        if phase in phase_times:
            phase_times[phase] += duration
        else:
            phase_times[phase] = duration

    total = elapsed[-1] - elapsed[0]

    # Phase 순서 정의
    phase_order = ["data_creation", "model_load", "model_to_cuda", "input_preparation", "vlm_inference"]
    display_names = {
        "data_creation":     "Phase 1: Data creation",
        "model_load":        "Phase 2+3: Model load (4-bit)",
        "model_to_cuda":     "Phase 2+3: Model .to(cuda)",
        "input_preparation": "Phase 4: Input preparation",
        "vlm_inference":     "Phase 5: VLM inference",
    }

    labels = []
    times = []
    pcts = []
    for phase in phase_order:
        if phase in phase_times:
            t = phase_times[phase]
            labels.append(display_names.get(phase, phase))
            times.append(round(t, 2))
            pcts.append(round(t / total * 100, 1))

    colors = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350"][:len(labels)]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5)

    for bar, t, p in zip(bars, times, pcts):
        w = bar.get_width()
        ax.text(
            w + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s ({p:.1f}%)",
            va="center", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("소요 시간 (초)" if FONT else "Time (s)", fontsize=11)
    ax.set_title(
        "4-bit Phase별 소요 시간 분석" if FONT else "4-bit Phase Execution Time Breakdown",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(times) * 1.3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    total_label = f"Total: {total:.2f}s"
    ax.text(
        0.98, 0.02, total_label,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontstyle="italic", color="#555555",
    )

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase_breakdown_4bit.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────
# Chart 3: FP16 vs 4-bit 비교 차트
# ──────────────────────────────────────────────
def chart_comparison(elapsed, allocated, phases):
    # 4-bit 결과 추출
    spans = get_phase_spans(elapsed, phases)
    phase_times = {}
    for start, end, phase in spans:
        duration = end - start
        if phase in phase_times:
            phase_times[phase] += duration
        else:
            phase_times[phase] = duration

    q4_model_load = phase_times.get("model_load", 0) + phase_times.get("model_to_cuda", 0)
    q4_inference = phase_times.get("vlm_inference", 0)
    q4_peak = max(allocated)
    q4_total = elapsed[-1] - elapsed[0]

    # FP16 베이스라인 값
    fp16_model_load = 54.2 + 56.86  # CPU load + to(cuda)
    fp16_inference = 273.79
    fp16_peak = 21.52
    fp16_total = 348.97

    # 비교 항목
    categories = [
        "모델 로드 시간\n(초)" if FONT else "Model Load\nTime (s)",
        "추론 시간\n(초)" if FONT else "Inference\nTime (s)",
        "Peak VRAM\n(GB)",
        "총 시간\n(초)" if FONT else "Total\nTime (s)",
    ]
    fp16_values = [fp16_model_load, fp16_inference, fp16_peak, fp16_total]
    q4_values = [q4_model_load, q4_inference, q4_peak, q4_total]

    x = np.arange(len(categories))
    width = 0.32

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_fp16 = ax.bar(x - width/2, fp16_values, width, label="FP16 (baseline)",
                        color="#EF5350", edgecolor="white", linewidth=0.8)
    bars_4bit = ax.bar(x + width/2, q4_values, width, label="4-bit (BnB fp4)",
                        color="#42A5F5", edgecolor="white", linewidth=0.8)

    # 각 바 위에 수치 라벨
    def add_bar_labels(bars, values):
        for bar, v in zip(bars, values):
            height = bar.get_height()
            label = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(fp16_values) * 0.015,
                label,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    add_bar_labels(bars_fp16, fp16_values)
    add_bar_labels(bars_4bit, q4_values)

    # 개선율 표시 (바 사이)
    for i in range(len(categories)):
        fp_v = fp16_values[i]
        q4_v = q4_values[i]
        reduction_pct = (1 - q4_v / fp_v) * 100
        if reduction_pct > 0:
            label = f"-{reduction_pct:.0f}%"
            color = "#2E7D32"
        else:
            label = f"+{-reduction_pct:.0f}%"
            color = "#C62828"
        y_pos = max(fp_v, q4_v) + max(fp16_values) * 0.06
        ax.text(
            x[i], y_pos, label,
            ha="center", va="bottom", fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.2, alpha=0.9),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("값" if FONT else "Value", fontsize=11)
    ax.set_title(
        "FP16 vs 4-bit 양자화 성능 비교" if FONT else "FP16 vs 4-bit Quantization Performance Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(fp16_values) * 1.25)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "comparison_fp16_vs_4bit.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Alpamayo 4-bit 양자화 실험 결과 시각화")
    print("=" * 50)

    elapsed, allocated, reserved, phases = load_csv(CSV_PATH)
    print(f"CSV 로드 완료: {len(elapsed)} 샘플")

    chart_memory_timeline(elapsed, allocated, reserved, phases)
    chart_phase_breakdown(elapsed, phases)
    chart_comparison(elapsed, allocated, phases)

    print("=" * 50)
    print("완료! 모든 차트가 figures/ 디렉토리에 저장되었습니다.")
