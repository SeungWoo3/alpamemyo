"""Alpamayo-R1-10B VRAM 점유 비율 시각화"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# === 1. 모델 파라미터 메모리 분포 (파이 차트) ===
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 모델 파라미터 breakdown
param_labels = ['VLM Language Model\n(7.58B, 15.17 GB)',
                'Expert/Decoder\n(2.28B, 4.56 GB)',
                'VLM LM Head\n(0.64B, 1.28 GB)',
                'Vision Encoder\n(0.58B, 1.15 GB)']
param_sizes = [15.17, 4.56, 1.28, 1.15]
param_colors = ['#e74c3c', '#3498db', '#e67e22', '#2ecc71']
param_pcts = [s/sum(param_sizes)*100 for s in param_sizes]

wedges, texts, autotexts = axes[0].pie(
    param_sizes, labels=param_labels, autopct='%1.1f%%',
    colors=param_colors, startangle=90, pctdistance=0.75,
    textprops={'fontsize': 10})
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
axes[0].set_title('Model Parameters (BF16)\nTotal: 22.16 GB', fontsize=14, fontweight='bold')

# === 2. 추론 시 전체 VRAM 점유 (스택 바 차트) ===
categories = ['Model\nParameters', 'KV Cache\n(~4,470 tokens)', 'Activation\nMemory', 'CUDA/PyTorch\nOverhead']
sizes = [22.16, 0.56, 0.8, 0.3]
colors = ['#e74c3c', '#9b59b6', '#f39c12', '#95a5a6']

bars = axes[1].barh(categories, sizes, color=colors, edgecolor='white', linewidth=1.5, height=0.6)

# 12GB VRAM 라인
axes[1].axvline(x=12, color='red', linestyle='--', linewidth=2, label='12GB VRAM Limit')
axes[1].axvline(x=21.52, color='orange', linestyle=':', linewidth=2, label='Peak Measured: 21.52 GB')

for bar, size in zip(bars, sizes):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{size:.2f} GB ({size/sum(sizes)*100:.1f}%)',
                va='center', fontsize=11, fontweight='bold')

axes[1].set_xlabel('Memory (GB)', fontsize=12)
axes[1].set_title('VRAM Occupancy During FP16 Inference\nPeak: ~21.52 GB', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 28)
axes[1].legend(fontsize=10, loc='lower right')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/seungwoo/workspace/seminar/figures/vram_breakdown.png', dpi=150, bbox_inches='tight')
print("Saved: seminar/figures/vram_breakdown.png")

# === 3. 모듈별 파라미터 + KV Cache 상세 분류 (스택 바) ===
fig2, ax2 = plt.subplots(figsize=(14, 8))

# 모듈별 상세 breakdown
modules = ['Vision\nEncoder', 'VLM\nLanguage\nModel', 'VLM\nLM Head', 'Expert\n(Decoder)', 'KV Cache', 'Activation\n+ Overhead']
mem_vals = [1.15, 15.17, 1.28, 4.56, 0.56, 1.1]
bar_colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6', '#95a5a6']

bars2 = ax2.bar(modules, mem_vals, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.65)

# 12GB 라인
ax2.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.7, label='12GB VRAM Limit')

# 값 표시
for bar, val in zip(bars2, mem_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.2f} GB\n({val/sum(mem_vals)*100:.1f}%)',
             ha='center', fontsize=11, fontweight='bold')

# 카테고리 구분선
ax2.axvline(x=3.5, color='gray', linestyle=':', alpha=0.5)
ax2.text(1.5, max(mem_vals) + 1.5, 'Static (Model Parameters)\n22.16 GB',
         ha='center', fontsize=11, color='#333', fontstyle='italic')
ax2.text(4.5, max(mem_vals) + 1.5, 'Dynamic\n(Per-Inference)\n1.66 GB',
         ha='center', fontsize=11, color='#333', fontstyle='italic')

ax2.set_ylabel('Memory (GB)', fontsize=13)
ax2.set_title('Alpamayo-R1-10B: Detailed VRAM Occupancy\nTotal Peak: ~21.52 GB (Static 93% + Dynamic 7%)',
              fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(mem_vals) + 3)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/seungwoo/workspace/seminar/figures/vram_detailed.png', dpi=150, bbox_inches='tight')
print("Saved: seminar/figures/vram_detailed.png")

# === 4. 스와핑 시나리오별 VRAM 사용 비교 ===
fig3, ax3 = plt.subplots(figsize=(14, 7))

scenarios = ['All on GPU\n(24GB+ needed)', 'Module Swap\n(Peak = VLM)', 'Layer Swap\n(BF16)',
             'INT4 + Module\n(Excluded)', 'Current\nBaseline']
# 각 시나리오에서의 구성요소별 VRAM
# [Vision, VLM, Expert, KV Cache, Activation, Other]
scenario_data = {
    'Vision': [1.15, 1.15, 0.39, 0.29, 1.15],
    'VLM': [15.17, 15.17, 0.39, 4.11, 15.17],
    'Expert': [4.56, 0, 0, 1.14, 4.56],
    'KV Cache': [0.56, 0.56, 0.56, 0.56, 0.56],
    'Other': [1.1, 1.1, 1.1, 1.1, 1.1],
}

bottom = np.zeros(5)
colors_stack = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#95a5a6']
for (label, vals), color in zip(scenario_data.items(), colors_stack):
    ax3.bar(scenarios, vals, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5, width=0.6)
    bottom += np.array(vals)

# 12GB 라인
ax3.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.8, label='12GB VRAM Limit')

# Peak 값 표시
peaks = [sum(v[i] for v in scenario_data.values()) for i in range(5)]
for i, peak in enumerate(peaks):
    ax3.text(i, peak + 0.3, f'{peak:.1f} GB', ha='center', fontsize=11, fontweight='bold')

ax3.set_ylabel('VRAM Usage (GB)', fontsize=13)
ax3.set_title('VRAM Usage by Loading Strategy\n(Only Layer-by-Layer BF16 fits in 12GB without quantization)',
              fontsize=14, fontweight='bold')
ax3.set_ylim(0, 28)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/seungwoo/workspace/seminar/figures/vram_scenarios.png', dpi=150, bbox_inches='tight')
print("Saved: seminar/figures/vram_scenarios.png")

plt.close('all')
print("\nAll 3 figures generated successfully!")
