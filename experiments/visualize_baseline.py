#!/usr/bin/env python3
"""Alpamayo-R1-10B 베이스라인 실험 결과 시각화 스크립트.

생성 차트:
  1. memory_timeline.png   - VRAM 사용량 시계열
  2. phase_breakdown.png   - Phase별 소요 시간 분석
  3. theory_vs_actual.png  - 이론 스왑 시간 vs 실제 추론 시간
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────
# 한국어 폰트 설정
# ──────────────────────────────────────────────
def setup_korean_font():
    """NanumGothic 등 한국어 폰트가 있으면 설정, 없으면 영어 fallback."""
    from matplotlib import font_manager
    korean_fonts = [
        "NanumGothic",
        "NanumBarunGothic",
        "Malgun Gothic",
        "AppleGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in korean_fonts:
        if font in available:
            plt.rcParams["font.family"] = font
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[폰트] '{font}' 사용")
            return font
    # fallback — 한국어 깨질 수 있음
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    print("[폰트] 한국어 폰트 없음 — DejaVu Sans fallback (한글 깨질 수 있음)")
    return None

FONT = setup_korean_font()

# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "memory_profile.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 150

# Phase 색상 맵 (axvspan 배경용, alpha 낮게)
PHASE_COLORS = {
    "init":              "#BDBDBD",
    "data_creation":     "#81D4FA",
    "model_load_cpu":    "#A5D6A7",
    "model_to_cuda":     "#FFCC80",
    "input_preparation": "#CE93D8",
    "vlm_inference":     "#EF9A9A",
}

PHASE_LABELS = {
    "init":              "Init",
    "data_creation":     "Data creation",
    "model_load_cpu":    "Model load (CPU)",
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
# Chart 1: VRAM 사용량 시계열
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
        "Alpamayo-R1-10B VRAM 사용량 (RTX 3080 Ti 12GB)" if FONT
        else "Alpamayo-R1-10B VRAM Usage (RTX 3080 Ti 12GB)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(elapsed[0], elapsed[-1])
    ax.set_ylim(0, max(reserved) * 1.08)

    # 범례 — 중복 제거
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
    out = os.path.join(FIG_DIR, "memory_timeline.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────
# Chart 2: Phase별 소요 시간 수평 막대
# ──────────────────────────────────────────────
def chart_phase_breakdown():
    # 프로파일링 결과 (문제 정의에서 제공)
    data = [
        ("Phase 1: Data creation",     0.08,   0.0),
        ("Phase 2: Model load (CPU)",  13.51,  3.9),
        ("Phase 3: Model .to(cuda)",   56.86, 16.3),
        ("Phase 4: Input preparation",  4.73,  1.4),
        ("Phase 5: VLM inference",    273.79, 78.5),
    ]
    labels = [d[0] for d in data]
    times = [d[1] for d in data]
    pcts = [d[2] for d in data]

    colors = ["#42A5F5", "#66BB6A", "#FFA726", "#AB47BC", "#EF5350"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, times, color=colors, edgecolor="white", linewidth=0.5)

    # 각 막대에 라벨
    for bar, t, p in zip(bars, times, pcts):
        w = bar.get_width()
        ax.text(
            w + 3, bar.get_y() + bar.get_height() / 2,
            f"{t:.2f}s ({p:.1f}%)",
            va="center", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("소요 시간 (초)" if FONT else "Time (s)", fontsize=11)
    ax.set_title(
        "Phase별 소요 시간 분석" if FONT else "Phase Execution Time Breakdown",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, max(times) * 1.25)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # 총 시간 표기
    total_label = f"Total: 348.97s"
    ax.text(
        0.98, 0.02, total_label,
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontstyle="italic", color="#555555",
    )

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "phase_breakdown.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────
# Chart 3: 이론 vs 실제 비교 (로그 스케일)
# ──────────────────────────────────────────────
def chart_theory_vs_actual():
    categories = [
        "이론 스왑 시간\n(NVMe → VRAM)" if FONT else "Theoretical Swap\n(NVMe -> VRAM)",
        "실제 VLM 추론 시간" if FONT else "Actual VLM Inference",
    ]
    values = [0.6, 273.79]
    colors = ["#42A5F5", "#EF5350"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, values, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("시간 (초, 로그 스케일)" if FONT else "Time (s, log scale)", fontsize=11)
    ax.set_title(
        "이론 스왑 시간 vs 실제 추론 시간" if FONT else "Theoretical Swap Time vs Actual Inference Time",
        fontsize=13, fontweight="bold",
    )

    # 값 라벨
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.4,
            f"{v}s",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    # 차이 배수 표시 (화살표 + 라벨)
    ratio = values[1] / values[0]
    mid_x = (bars[0].get_x() + bars[0].get_width() / 2 +
             bars[1].get_x() + bars[1].get_width() / 2) / 2
    mid_y = (values[0] * values[1]) ** 0.5  # 로그 스케일의 기하 평균 위치

    ax.annotate(
        f"~{ratio:.0f}x 차이" if FONT else f"~{ratio:.0f}x gap",
        xy=(mid_x, mid_y),
        fontsize=14, fontweight="bold", color="#D32F2F",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D32F2F", lw=1.5),
    )

    # y축 포맷
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_ylim(0.1, 1000)
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "theory_vs_actual.png")
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[저장] {out}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Alpamayo 베이스라인 실험 결과 시각화")
    print("=" * 50)

    elapsed, allocated, reserved, phases = load_csv(CSV_PATH)
    print(f"CSV 로드 완료: {len(elapsed)} 샘플")

    chart_memory_timeline(elapsed, allocated, reserved, phases)
    chart_phase_breakdown()
    chart_theory_vs_actual()

    print("=" * 50)
    print("완료! 모든 차트가 figures/ 디렉토리에 저장되었습니다.")
