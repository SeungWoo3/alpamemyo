import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
R1_DIR = os.path.dirname(BASE_DIR)
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


results = load_json(os.path.join(BASE_DIR, "results.json"))
vram = load_json(os.path.join(BASE_DIR, "vram_timeline.json"))

# 1) VRAM 타임라인
fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
dark_style(ax)
ax.plot(vram["timestamps"], vram["allocated_gb"], color='#00bcd4', linewidth=1.5, label='VRAM Allocated')
ax.plot(vram["timestamps"], vram["reserved_gb"], color='#ff9800', linewidth=1, alpha=0.7, label='VRAM Reserved')
ax.axhline(y=16, color='#f44336', linestyle='--', alpha=0.8, label='16GB VRAM 한계')

ml_start = results.get("model_load_start_t", 0)
ml_end = results.get("model_load_end_t", 0)
inf_start = results.get("inference_start_t", 0)
inf_end = results.get("inference_end_t", 0)
if ml_start and ml_end:
    ax.axvspan(ml_start, ml_end, alpha=0.15, color='#4caf50', label=f'모델 로딩+셋업 ({results.get("setup_time", 0)}s)')
if inf_start and inf_end:
    ax.axvspan(inf_start, inf_end, alpha=0.15, color='#2196f3', label=f'추론 ({results.get("inference_time", 0)}s)')

ax.set_xlabel('시간 (초)')
ax.set_ylabel('VRAM 사용량 (GB)')
ax.set_title('5070 Ti On-Demand Layering - VRAM 타임라인\n(Pinned Memory, No Prefetch, 25L offload)', fontsize=13)
ax.legend(loc='upper left', fontsize=9, facecolor=DARK_BG, edgecolor=DARK_GRID, labelcolor=DARK_FG)
ax.set_ylim(0, 18)

stats = results.get("transfer_stats", {})
h2d_total = stats.get('h2d_total_s', 0)
free_total = stats.get('free_total_s', 0)
compute = results.get('inference_time', 0) - h2d_total - free_total if results.get('inference_time') else 0
info = (f"추론: {results.get('inference_time', 'N/A')}s\n"
        f"H2D: {stats.get('h2d_total_s', 'N/A')}s (avg {stats.get('h2d_avg_ms', 'N/A')}ms)\n"
        f"FREE: {stats.get('free_total_s', 'N/A')}s\n"
        f"연산: {compute:.1f}s\n"
        f"Peak VRAM: {results.get('peak_vram_gb', 'N/A')} GB\n"
        f"오프로드: {results.get('vlm_layers_offloaded', 'N/A')}/36 layers")
ax.text(0.98, 0.70, info, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='#2d2d2d', alpha=0.9, edgecolor=DARK_GRID),
        color=DARK_FG)

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "vram_timeline.png"), dpi=150, facecolor=DARK_BG)
plt.close()
print("  on-demand-layering/vram_timeline.png 저장")

# 2) 전체 실험 비교 바 차트
experiments = []

# baseline
try:
    bl = load_json(os.path.join(R1_DIR, "baseline", "baseline_results.json"))
    experiments.append(("Baseline\n(device_map=auto)", bl["inference_time"], '#42a5f5'))
except Exception as e:
    print(f"  baseline 로드 실패: {e}")

# max-clock
try:
    mc = load_json(os.path.join(R1_DIR, "max-clock", "baseline_maxclock_results.json"))
    experiments.append(("Max Clock\n(device_map=auto)", mc["inference_time"], '#66bb6a'))
except Exception as e:
    print(f"  max-clock 로드 실패: {e}")

# pinned-memory
try:
    pm = load_json(os.path.join(R1_DIR, "pinned-memory", "results.json"))
    experiments.append(("Pinned Memory\n(25L offload)", pm["inference_time"], '#ab47bc'))
except Exception as e:
    print(f"  pinned-memory 로드 실패: {e}")

# on-demand-layering (이번 실험)
experiments.append(("On-Demand Layering\n(Pinned, 25L offload)", results["inference_time"], '#ff7043'))

fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
dark_style(ax)

labels = [e[0] for e in experiments]
vals = [e[1] for e in experiments]
colors = [e[2] for e in experiments]

bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.1f}s', ha='center', va='bottom', color=DARK_FG, fontsize=11, fontweight='bold')

# 속도향상 표시
if len(experiments) >= 1:
    baseline_val = experiments[0][1]
    for bar, val in zip(bars, vals):
        speedup = baseline_val / val
        if speedup > 1.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                    f'{speedup:.1f}x', ha='center', va='center', color='white', fontsize=10, fontweight='bold')

ax.set_ylabel('추론 시간 (s)')
ax.set_title('5070 Ti 전체 실험 비교 — 추론 시간', fontsize=14, fontweight='bold', color=DARK_FG)

plt.tight_layout()
fig.savefig(os.path.join(BASE_DIR, "all_experiments_comparison.png"), dpi=150, facecolor=DARK_BG)
plt.close()

print("  on-demand-layering/all_experiments_comparison.png 저장")
print("\n시각화 완료!")

# 요약 출력
print(f"\n=== 핵심 결과 ===")
print(f"추론 시간: {results['inference_time']}s")
print(f"H2D avg: {stats.get('h2d_avg_ms')}ms/layer")
print(f"Peak VRAM: {results.get('peak_vram_gb')} GB")
print(f"minADE: {results.get('minADE', 'N/A')}m")
