"""
R1-5070ti 실험 전체 시각화 (한국어 폰트 포함)
baseline, max-clock 결과를 플롯으로 생성
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 한국어 폰트 설정
matplotlib.rcParams['font.family'] = 'NanumGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DARK_BG = '#1e1e1e'
DARK_FG = '#e0e0e0'
DARK_GRID = '#333333'


def dark_style(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=DARK_FG)
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)
    for spine in ax.spines.values():
        spine.set_color(DARK_GRID)
    ax.grid(True, alpha=0.3, color=DARK_GRID)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_baseline_vram():
    """baseline VRAM 타임라인"""
    vram = load_json(os.path.join(BASE_DIR, "baseline", "baseline_vram_timeline.json"))
    results = load_json(os.path.join(BASE_DIR, "baseline", "baseline_results.json"))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    dark_style(ax)

    ax.plot(vram["timestamps"], vram["allocated_gb"], color='#00bcd4', linewidth=1.5, label='VRAM Allocated (GB)')
    ax.plot(vram["timestamps"], vram["reserved_gb"], color='#ff9800', linewidth=1, alpha=0.7, label='VRAM Reserved (GB)')
    ax.axhline(y=16, color='#f44336', linestyle='--', alpha=0.8, label='16GB VRAM 한계')

    # 모델 로딩 / 추론 구간 표시
    ml_start = results.get("model_load_start_t", 0)
    ml_end = results.get("model_load_end_t", 0)
    inf_start = results.get("inference_start_t", 0)
    inf_end = results.get("inference_end_t", 0)

    if ml_start and ml_end:
        ax.axvspan(ml_start, ml_end, alpha=0.15, color='#4caf50', label=f'모델 로딩 ({results["model_load_time"]:.1f}s)')
    if inf_start and inf_end:
        ax.axvspan(inf_start, inf_end, alpha=0.15, color='#2196f3', label=f'추론 ({results["inference_time"]:.1f}s)')

    ax.set_xlabel('시간 (초)')
    ax.set_ylabel('VRAM 사용량 (GB)')
    ax.set_title('5070 Ti 베이스라인 - VRAM 타임라인\n(Alpamayo-R1-10B, device_map="auto")', fontsize=13)
    ax.legend(loc='upper left', fontsize=9, facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)
    ax.set_ylim(0, 18)

    # 정보 텍스트
    info = (f"Peak VRAM: {results['peak_vram_allocated']:.2f} GB\n"
            f"추론 시간: {results['inference_time']:.1f}s\n"
            f"minADE: {results['minADE']:.4f}m")
    ax.text(0.98, 0.75, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.9, edgecolor=DARK_GRID),
            color=DARK_FG)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "baseline", "vram_timeline.png")
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  저장: {out}")


def plot_maxclock_comparison():
    """기본 vs 클럭고정 비교"""
    baseline = load_json(os.path.join(BASE_DIR, "baseline", "baseline_results.json"))
    maxclock = load_json(os.path.join(BASE_DIR, "max-clock", "baseline_maxclock_results.json"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=DARK_BG)

    labels = ['기본\n(자동 클럭)', '클럭고정\n(3090/14001 MHz)']
    colors = ['#42a5f5', '#66bb6a']

    # 1) 추론 시간
    ax = axes[0]
    dark_style(ax)
    vals = [baseline["inference_time"], maxclock["inference_time"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}s', ha='center', va='bottom', color=DARK_FG, fontsize=12, fontweight='bold')
    speedup = baseline["inference_time"] / maxclock["inference_time"]
    ax.set_title(f'추론 시간 (s)\n클럭고정 {speedup:.2f}x 개선', fontsize=12)
    ax.set_ylabel('시간 (s)')

    # 2) 모델 로드 시간
    ax = axes[1]
    dark_style(ax)
    vals = [baseline["model_load_time"], maxclock["model_load_time"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', color=DARK_FG, fontsize=12, fontweight='bold')
    ax.set_title('모델 로드 시간 (s)', fontsize=12)
    ax.set_ylabel('시간 (s)')

    # 3) Peak VRAM
    ax = axes[2]
    dark_style(ax)
    vals = [baseline["peak_vram_allocated"], maxclock["peak_vram_allocated"]]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.axhline(y=16, color='#f44336', linestyle='--', alpha=0.8, label='16GB VRAM 한계')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}GB', ha='center', va='bottom', color=DARK_FG, fontsize=12, fontweight='bold')
    ax.set_title('Peak VRAM Allocated', fontsize=12)
    ax.set_ylabel('VRAM (GB)')
    ax.legend(facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)

    fig.suptitle('RTX 5070 Ti - 기본 vs 클럭고정 추론 비교\n(Alpamayo-R1-10B, device_map="auto")',
                 fontsize=14, color=DARK_FG, fontweight='bold')

    # 하단 정보
    info = ('【기본】 Graphics: 180 MHz (P8→P0 자동)\n'
            '【클럭고정】 Graphics: 3090 MHz, Mem: 14001 MHz\n'
            'PCIe: Gen5 x16 (양쪽 동일)\n'
            f'minADE: {baseline["minADE"]:.4f}m (양쪽 동일 → 재현성 확보)')
    fig.text(0.5, -0.02, info, ha='center', fontsize=10, color='#aaaaaa',
             bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.9, edgecolor=DARK_GRID))

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "max-clock", "inference_comparison.png")
    fig.savefig(out, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close()
    print(f"  저장: {out}")


def plot_maxclock_vram_comparison():
    """기본 vs 클럭고정 VRAM 타임라인 비교"""
    vram_base = load_json(os.path.join(BASE_DIR, "baseline", "baseline_vram_timeline.json"))
    vram_max = load_json(os.path.join(BASE_DIR, "max-clock", "baseline_maxclock_vram_timeline.json"))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    dark_style(ax)

    ax.plot(vram_base["timestamps"], vram_base["allocated_gb"],
            color='#42a5f5', linewidth=1.5, label='기본 (자동 클럭)', alpha=0.8)
    ax.plot(vram_max["timestamps"], vram_max["allocated_gb"],
            color='#66bb6a', linewidth=1.5, label='클럭고정 (3090/14001 MHz)', alpha=0.8)
    ax.axhline(y=16, color='#f44336', linestyle='--', alpha=0.8, label='16GB VRAM 한계')

    ax.set_xlabel('시간 (초)')
    ax.set_ylabel('VRAM Allocated (GB)')
    ax.set_title('5070 Ti VRAM 타임라인 비교 - 기본 vs 클럭고정', fontsize=13)
    ax.legend(facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)
    ax.set_ylim(0, 18)

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "max-clock", "vram_comparison.png")
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  저장: {out}")


if __name__ == "__main__":
    print("R1-5070ti 시각화 생성 중...")
    plot_baseline_vram()
    plot_maxclock_comparison()
    plot_maxclock_vram_comparison()
    print("완료!")
