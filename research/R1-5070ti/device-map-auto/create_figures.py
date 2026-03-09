"""
5070ti 베이스라인 결과 시각화
- VRAM 사용량 시계열 플롯
- 로딩/추론 구간 구분 표시
- 기본 vs 클럭고정 비교 플롯
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 폰트 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = "/home/seungwoo/workspace/research/R1-5070ti/baseline"

# ── 데이터 로드 ─────────────────────────────────────────────
with open(f"{BASE_DIR}/baseline_results.json") as f:
    results_base = json.load(f)

with open(f"{BASE_DIR}/baseline_vram_timeline.json") as f:
    vram_base = json.load(f)

with open(f"{BASE_DIR}/baseline_maxclock_results.json") as f:
    results_max = json.load(f)

with open(f"{BASE_DIR}/baseline_maxclock_vram_timeline.json") as f:
    vram_max = json.load(f)

ts_base = np.array(vram_base["timestamps"])
alloc_base = np.array(vram_base["allocated_gb"])
resv_base = np.array(vram_base["reserved_gb"])

ts_max = np.array(vram_max["timestamps"])
alloc_max = np.array(vram_max["allocated_gb"])
resv_max = np.array(vram_max["reserved_gb"])


# ═══════════════════════════════════════════════════════════
# Figure 1: 기존 베이스라인 VRAM Timeline
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0f0f0f")
ax.set_facecolor("#1a1a1a")

ax.plot(ts_base, resv_base, color="#4a9eff", linewidth=1.5,
        alpha=0.6, label="VRAM Reserved (GB)")
ax.plot(ts_base, alloc_base, color="#00e5ff", linewidth=2.0,
        label="VRAM Allocated (GB)")

ax.axhline(y=16.0, color="#ff4444", linewidth=1.5, linestyle="--",
           alpha=0.8, label="16GB VRAM Limit")

model_start = results_base["model_load_start_t"]
model_end = results_base["model_load_end_t"]
ax.axvspan(model_start, model_end, alpha=0.15, color="#ffaa00",
           label=f"Model Loading ({results_base['model_load_time']:.1f}s)")
ax.axvline(x=model_start, color="#ffaa00", linewidth=1.2, linestyle=":", alpha=0.8)
ax.axvline(x=model_end, color="#ffaa00", linewidth=1.2, linestyle=":", alpha=0.8)

infer_start = results_base["inference_start_t"]
infer_end = results_base["inference_end_t"]
ax.axvspan(infer_start, infer_end, alpha=0.15, color="#44ff88",
           label=f"Inference ({results_base['inference_time']:.1f}s)")
ax.axvline(x=infer_start, color="#44ff88", linewidth=1.2, linestyle=":", alpha=0.8)
ax.axvline(x=infer_end, color="#44ff88", linewidth=1.2, linestyle=":", alpha=0.8)

ax.text((model_start + model_end) / 2, 0.4, "Model\nLoading",
        ha="center", va="bottom", color="#ffaa00", fontsize=9, fontweight="bold")
ax.text((infer_start + infer_end) / 2, 0.4, "Inference",
        ha="center", va="bottom", color="#44ff88", fontsize=9, fontweight="bold")

summary = (
    f"GPU: RTX 5070 Ti 16GB\n"
    f"Method: device_map='auto'\n"
    f"  (*.to('cuda') -> OOM)\n\n"
    f"Model Load:  {results_base['model_load_time']:.2f}s\n"
    f"Inference:   {results_base['inference_time']:.2f}s\n"
    f"Total:       {results_base['total_time']:.2f}s\n\n"
    f"Peak VRAM:   {results_base['peak_vram_allocated']:.2f} GB\n"
    f"minADE:      {results_base['minADE']:.4f}m\n\n"
    f"Device: CUDA 649 layers\n"
    f"        CPU  510 layers (meta)"
)
ax.text(0.02, 0.97, summary, transform=ax.transAxes,
        fontsize=8.5, verticalalignment="top", family="monospace",
        color="#e0e0e0",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2a2a",
                  edgecolor="#555555", alpha=0.9))

ax.set_xlabel("Time (s)", color="#cccccc", fontsize=11)
ax.set_ylabel("VRAM Usage (GB)", color="#cccccc", fontsize=11)
ax.set_title("5070ti Baseline Inference - VRAM Timeline\n"
             "(Alpamayo-R1-10B, device_map='auto')",
             color="#ffffff", fontsize=13, fontweight="bold", pad=12)

ax.set_xlim(0, ts_base[-1] * 1.02)
ax.set_ylim(0, 17)
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")

ax.legend(loc="upper right", fontsize=9,
          facecolor="#2a2a2a", edgecolor="#555555",
          labelcolor="#dddddd")
ax.grid(True, color="#333333", linewidth=0.5, alpha=0.7)

plt.tight_layout()
out_path = f"{BASE_DIR}/vram_timeline.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"저장: {out_path}")
plt.close()


# ═══════════════════════════════════════════════════════════
# Figure 2: VRAM Timeline 비교 (기본 vs 클럭고정)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("RTX 5070 Ti - 기본 vs 클럭고정 VRAM Timeline 비교\n"
             "(Alpamayo-R1-10B, device_map='auto')",
             color="#ffffff", fontsize=14, fontweight="bold", y=0.98)

plot_configs = [
    (axes[0], ts_base, alloc_base, resv_base, results_base,
     "기본 (아이들 클럭, P8 → P0)", "#00e5ff", "#4a9eff"),
    (axes[1], ts_max, alloc_max, resv_max, results_max,
     "클럭 고정 (3090MHz / 14001MHz, P0)", "#44ff88", "#2aaa55"),
]

for ax, ts, alloc, resv, res, title, c_alloc, c_resv in plot_configs:
    ax.set_facecolor("#1a1a1a")
    ax.plot(ts, resv, color=c_resv, linewidth=1.5, alpha=0.6, label="VRAM Reserved")
    ax.plot(ts, alloc, color=c_alloc, linewidth=2.0, label="VRAM Allocated")
    ax.axhline(y=16.0, color="#ff4444", linewidth=1.2, linestyle="--",
               alpha=0.7, label="16GB Limit")

    # 구간 표시
    ms = res["model_load_start_t"]
    me = res["model_load_end_t"]
    is_ = res["inference_start_t"]
    ie = res["inference_end_t"]

    ax.axvspan(ms, me, alpha=0.12, color="#ffaa00")
    ax.axvspan(is_, ie, alpha=0.12, color=c_alloc)
    ax.axvline(x=ms, color="#ffaa00", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.axvline(x=me, color="#ffaa00", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.axvline(x=is_, color=c_alloc, linewidth=1.0, linestyle=":", alpha=0.7)
    ax.axvline(x=ie, color=c_alloc, linewidth=1.0, linestyle=":", alpha=0.7)

    ax.text((ms + me) / 2, 0.5, f"로딩\n{res['model_load_time']:.1f}s",
            ha="center", va="bottom", color="#ffaa00", fontsize=8.5, fontweight="bold")
    ax.text((is_ + ie) / 2, 0.5, f"추론\n{res['inference_time']:.1f}s",
            ha="center", va="bottom", color=c_alloc, fontsize=8.5, fontweight="bold")

    ax.set_title(title, color="#ffffff", fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("VRAM (GB)", color="#cccccc", fontsize=10)
    ax.set_ylim(0, 17)
    ax.set_xlim(0, ts[-1] * 1.02)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.legend(loc="upper right", fontsize=8.5,
              facecolor="#2a2a2a", edgecolor="#555555", labelcolor="#dddddd")
    ax.grid(True, color="#333333", linewidth=0.5, alpha=0.7)

axes[1].set_xlabel("Time (s)", color="#cccccc", fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = f"{BASE_DIR}/vram_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"저장: {out_path}")
plt.close()


# ═══════════════════════════════════════════════════════════
# Figure 3: 추론 시간 비교 바 차트
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 6))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("RTX 5070 Ti - 기본 vs 클럭고정 성능 비교\n"
             "(Alpamayo-R1-10B, device_map='auto')",
             color="#ffffff", fontsize=14, fontweight="bold")

labels = ["기본\n(아이들 클럭)", "클럭고정\n(3090/14001 MHz)"]
colors_base = ["#4a9eff", "#44ff88"]

# 서브플롯 1: 추론 시간
ax = axes[0]
ax.set_facecolor("#1a1a1a")
vals = [results_base["inference_time"], results_max["inference_time"]]
bars = ax.bar(labels, vals, color=colors_base, width=0.5, edgecolor="#555555")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.1f}s", ha="center", va="bottom", color="#ffffff",
            fontsize=11, fontweight="bold")
speedup = vals[0] / vals[1]
ax.set_title(f"추론 시간 (s)\n클럭고정 {speedup:.2f}x 빠름",
             color="#ffffff", fontsize=10, fontweight="bold")
ax.set_ylabel("시간 (s)", color="#cccccc")
ax.set_ylim(0, max(vals) * 1.2)
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.grid(True, axis="y", color="#333333", linewidth=0.5, alpha=0.7)

# 서브플롯 2: 모델 로드 시간
ax = axes[1]
ax.set_facecolor("#1a1a1a")
vals = [results_base["model_load_time"], results_max["model_load_time"]]
bars = ax.bar(labels, vals, color=colors_base, width=0.5, edgecolor="#555555")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}s", ha="center", va="bottom", color="#ffffff",
            fontsize=11, fontweight="bold")
ax.set_title("모델 로드 시간 (s)",
             color="#ffffff", fontsize=10, fontweight="bold")
ax.set_ylabel("시간 (s)", color="#cccccc")
ax.set_ylim(0, max(vals) * 1.2)
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.grid(True, axis="y", color="#333333", linewidth=0.5, alpha=0.7)

# 서브플롯 3: Peak VRAM
ax = axes[2]
ax.set_facecolor("#1a1a1a")
vram_key_base = results_base.get("peak_vram_allocated", 0)
vram_key_max = results_max.get("peak_vram_allocated", 0)
vals = [vram_key_base, vram_key_max]
bars = ax.bar(labels, vals, color=colors_base, width=0.5, edgecolor="#555555")
ax.axhline(y=16.0, color="#ff4444", linewidth=1.5, linestyle="--",
           alpha=0.8, label="16GB Limit")
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.2f}GB", ha="center", va="bottom", color="#ffffff",
            fontsize=11, fontweight="bold")
ax.set_title("Peak VRAM Allocated",
             color="#ffffff", fontsize=10, fontweight="bold")
ax.set_ylabel("VRAM (GB)", color="#cccccc")
ax.set_ylim(0, 17)
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.legend(loc="upper right", fontsize=9,
          facecolor="#2a2a2a", edgecolor="#555555", labelcolor="#dddddd")
ax.grid(True, axis="y", color="#333333", linewidth=0.5, alpha=0.7)

# 클럭 정보 추가 텍스트
info_text = (
    f"[기본]  Graphics: 180 MHz (P8→P0 전환)\n"
    f"[클럭고정]  Graphics: 3090 MHz, Mem: 14001 MHz\n"
    f"PCIe: Gen5 x16 (양쪽 동일)\n\n"
    f"minADE: {results_base['minADE']:.4f}m (양쪽 동일 — 재현성 확인)"
)
fig.text(0.5, 0.02, info_text, ha="center", va="bottom",
         fontsize=9, color="#aaaaaa", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e1e",
                   edgecolor="#444444", alpha=0.9))

plt.tight_layout(rect=[0, 0.12, 1, 0.94])
out_path = f"{BASE_DIR}/inference_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print(f"저장: {out_path}")
plt.close()

print("\n모든 시각화 생성 완료.")
