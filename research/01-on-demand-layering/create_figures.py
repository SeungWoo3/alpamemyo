#!/usr/bin/env python3
"""Part 3: Create visualizations for On-Demand Layering analysis.

Generates:
1. Layer-by-layer memory distribution chart (stacked bar + pie)
2. device_map experiment VRAM analysis
3. FP16 baseline vs device_map vs manual offload comparison
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Korean font setup
FONT_PATH = "/home/seungwoo/.local/share/fonts/NanumGothic-Regular.ttf"
FONT_PATH_BOLD = "/home/seungwoo/.local/share/fonts/NanumGothic-Bold.ttf"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    if os.path.exists(FONT_PATH_BOLD):
        fm.fontManager.addfont(FONT_PATH_BOLD)
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Korean font loaded: {font_prop.get_name()}")
else:
    print(f"Warning: Korean font not found at {FONT_PATH}")

OUTPUT_DIR = "/home/seungwoo/workspace/research/01-on-demand-layering"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Colors
COLORS = {
    'vision': '#4CAF50',      # Green
    'vlm_lm': '#2196F3',      # Blue
    'vlm_head': '#03A9F4',    # Light Blue
    'expert': '#FF9800',      # Orange
    'diffusion': '#9C27B0',   # Purple
    'action_proj': '#F44336', # Red
    'other': '#9E9E9E',       # Gray
    'gpu': '#43A047',         # Dark Green
    'cpu': '#1E88E5',         # Dark Blue
    'disk': '#E53935',        # Dark Red
}

# ─── Load data ────────────────────────────────────────────────────────────────

with open(os.path.join(OUTPUT_DIR, "layer_analysis.json")) as f:
    layer_data = json.load(f)

with open(os.path.join(OUTPUT_DIR, "device_map_results.json")) as f:
    device_map_data = json.load(f)

with open(os.path.join(OUTPUT_DIR, "manual_offload_results.json")) as f:
    offload_data = json.load(f)


# ─── Figure 1: Memory Distribution ──────────────────────────────────────────

def create_memory_distribution():
    """Create stacked bar chart and pie chart for memory distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Data
    semantic = layer_data["semantic_groups"]
    components = {
        "Vision Encoder": semantic["Vision Encoder (vlm.visual)"]["memory_bytes"] / 1e9,
        "VLM Language\nModel": semantic["VLM Language Model (vlm.model - visual)"]["memory_bytes"] / 1e9,
        "VLM LM Head": semantic["VLM LM Head"]["memory_bytes"] / 1e9,
        "Expert\n(Trajectory\nDecoder)": semantic["Expert (Trajectory Decoder Backbone)"]["memory_bytes"] / 1e9,
        "Action\nProjections": (semantic["Action Projection (action_in_proj)"]["memory_bytes"] +
                                semantic["Action Projection (action_out_proj)"]["memory_bytes"]) / 1e9,
    }

    colors = [COLORS['vision'], COLORS['vlm_lm'], COLORS['vlm_head'],
              COLORS['expert'], COLORS['action_proj']]

    # Stacked bar chart
    ax1 = axes[0]
    bottom = 0
    bars = []
    for (name, size), color in zip(components.items(), colors):
        bar = ax1.bar(["Alpamayo R1\n(11.08B params)"], [size], bottom=[bottom],
                      color=color, label=f"{name} ({size:.2f} GB)", edgecolor='white', linewidth=0.5)
        # Add text label if segment is large enough
        if size > 0.5:
            ax1.text(0, bottom + size/2, f"{size:.2f} GB", ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        bottom += size

    ax1.set_ylabel("메모리 사용량 (GB)", fontsize=12)
    ax1.set_title("Alpamayo R1 서브모듈별 메모리 분포 (FP16/BF16)", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 25)
    ax1.axhline(y=12, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(0.5, 12.3, "RTX 3080 Ti VRAM (12 GB)", ha='center', fontsize=9, color='red')
    ax1.grid(axis='y', alpha=0.3)

    # Pie chart
    ax2 = axes[1]
    sizes = list(components.values())
    labels = [f"{name}\n{size:.2f} GB ({size/sum(sizes)*100:.1f}%)"
              for name, size in components.items() if size > 0.01]
    sizes_nonzero = [s for s in sizes if s > 0.01]
    colors_nonzero = [c for s, c in zip(sizes, colors) if s > 0.01]

    wedges, texts, autotexts = ax2.pie(
        sizes_nonzero,
        labels=None,
        colors=colors_nonzero,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
    )

    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax2.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax2.set_title("구성 비율", fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "01_memory_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 2: device_map VRAM Analysis ─────────────────────────────────────

def create_device_map_analysis():
    """Create device_map experiment analysis visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Device placement breakdown
    ax1 = axes[0]

    # Data from device_map results
    placement = device_map_data["device_placement_summary"]
    modules = []
    gpu_vals = []
    cpu_vals = []
    for mod, info in placement.items():
        if info["gpu_params_M"] + info["cpu_params_M"] > 1:  # Skip tiny modules
            modules.append(mod)
            gpu_vals.append(info["gpu_params_M"] / 1000)  # Convert to billions
            cpu_vals.append(info["cpu_params_M"] / 1000)

    x = np.arange(len(modules))
    width = 0.35

    bars_gpu = ax1.bar(x - width/2, gpu_vals, width, label='GPU', color=COLORS['gpu'], edgecolor='white')
    bars_cpu = ax1.bar(x + width/2, cpu_vals, width, label='CPU', color=COLORS['cpu'], edgecolor='white')

    ax1.set_ylabel("파라미터 수 (B)", fontsize=12)
    ax1.set_title("device_map='auto' 디바이스 배치", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modules, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars_gpu:
        if bar.get_height() > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{bar.get_height():.2f}B', ha='center', va='bottom', fontsize=9)
    for bar in bars_cpu:
        if bar.get_height() > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{bar.get_height():.2f}B', ha='center', va='bottom', fontsize=9)

    # Right: VLM layer-level placement
    ax2 = axes[1]

    # VLM has 36 transformer layers (0-35), layers 0-16 on GPU, 17-35 on CPU
    n_layers = 36
    layer_nums = list(range(n_layers))
    layer_device = ['GPU' if i <= 16 else 'CPU' for i in range(n_layers)]
    layer_colors = [COLORS['gpu'] if d == 'GPU' else COLORS['cpu'] for d in layer_device]
    layer_size_mb = [192.9] * n_layers  # Each ~192.9M params

    ax2.barh(layer_nums, layer_size_mb, color=layer_colors, edgecolor='white', linewidth=0.3)
    ax2.set_xlabel("파라미터 수 (M)", fontsize=12)
    ax2.set_ylabel("VLM Transformer 레이어 번호", fontsize=12)
    ax2.set_title("VLM 레이어별 GPU/CPU 배치", fontsize=14, fontweight='bold')
    ax2.invert_yaxis()

    # Add dividing line
    ax2.axhline(y=16.5, color='red', linestyle='--', linewidth=1.5)
    ax2.text(100, 8, "GPU (레이어 0-16)", fontsize=11, ha='center',
             color=COLORS['gpu'], fontweight='bold')
    ax2.text(100, 26, "CPU (레이어 17-35)", fontsize=11, ha='center',
             color=COLORS['cpu'], fontweight='bold')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['gpu'], label='GPU'),
                       Patch(facecolor=COLORS['cpu'], label='CPU')]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "02_device_map_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 3: Comparison Chart ─────────────────────────────────────────────

def create_comparison_chart():
    """Create comparison chart: FP16 baseline vs device_map vs manual offload scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # ── Left: VRAM comparison ──
    ax1 = axes[0]

    scenarios = [
        "FP16 Baseline\n(전체 GPU 로딩)",
        "device_map\n='auto'",
        "시나리오 A\n(레이어별 로딩)",
        "시나리오 B\n(모듈별 로딩)",
        "시나리오 C\n(INT4 + 순차)",
    ]

    # VRAM values
    vram_values = [
        21.52,  # FP16 baseline from prior work
        device_map_data["peak_vram_pytorch_gb"],  # device_map
        offload_data["on_demand_layering_analysis"]["scenario_a_theoretical_min_vram_gb"],  # Scenario A
        offload_data["on_demand_layering_analysis"]["scenario_b_peak_vram_gb"],  # Scenario B
        offload_data["on_demand_layering_analysis"]["scenario_c_vlm_int4_vram_gb"],  # Scenario C
    ]

    bar_colors = ['#E53935', '#FF9800', '#4CAF50', '#F44336', '#2196F3']
    feasibility = ['불가능\n(OOM)', '제한적\n(추론 실패)', '이론적\n가능', '불가능\n(OOM)', '가능']
    bar_hatches = ['', '', '//', '', '']

    bars = ax1.bar(range(len(scenarios)), vram_values, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Add pattern for theoretical
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)

    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels(scenarios, fontsize=9, ha='center')
    ax1.set_ylabel("Peak VRAM (GB)", fontsize=12)
    ax1.set_title("피크 VRAM 사용량 비교", fontsize=14, fontweight='bold')

    # 12GB line
    ax1.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(4.4, 12.3, "12 GB 제한", ha='right', fontsize=10, color='red', fontweight='bold')

    # Value labels
    for i, (bar, val, feas) in enumerate(zip(bars, vram_values, feasibility)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f} GB\n({feas})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylim(0, 25)
    ax1.grid(axis='y', alpha=0.3)

    # ── Right: Comprehensive comparison table ──
    ax2 = axes[1]
    ax2.axis('off')

    # Table data
    table_data = [
        ["전략", "피크 VRAM\n(GB)", "지연\n(예상)", "실현성", "메모리\n절감율"],
        ["FP16 Baseline", "21.52", "273.79s\n(기준)", "불가능\n(12GB 초과)", "-"],
        ["device_map\n='auto'", f"{device_map_data['peak_vram_pytorch_gb']:.2f}",
         "측정 불가\n(추론 실패)", "제한적\n(커스텀 토큰\n처리 오류)", f"{(1-device_map_data['peak_vram_pytorch_gb']/21.52)*100:.0f}%"],
        ["Scenario A:\n레이어별 로딩", f"{offload_data['on_demand_layering_analysis']['scenario_a_theoretical_min_vram_gb']:.2f}",
         "+15~30%\n(이론)", "높음\n(구현 복잡)", f"{offload_data['on_demand_layering_analysis']['scenario_a_memory_reduction_pct']:.0f}%"],
        ["Scenario B:\n모듈별 로딩", f"{offload_data['on_demand_layering_analysis']['scenario_b_peak_vram_gb']:.2f}",
         "+모듈 전환\n시간", "불가능\n(VLM > 12GB)", f"{(1-offload_data['on_demand_layering_analysis']['scenario_b_peak_vram_gb']/21.52)*100:.0f}%"],
        ["Scenario C:\nINT4 + 순차", f"{offload_data['on_demand_layering_analysis']['scenario_c_vlm_int4_vram_gb']:.2f}",
         "+20~40%\n(모델 전환)", "매우 높음\n(권장)", f"{(1-offload_data['on_demand_layering_analysis']['scenario_c_vlm_int4_vram_gb']/21.52)*100:.0f}%"],
    ]

    # Color coding
    cell_colors = [['#E0E0E0'] * 5]  # Header
    for row in table_data[1:]:
        row_colors = ['#F5F5F5'] * 5
        # Color VRAM column based on feasibility
        vram_val = float(row[1])
        if vram_val <= 12:
            row_colors[1] = '#C8E6C9'  # Green
        else:
            row_colors[1] = '#FFCDD2'  # Red
        cell_colors.append(row_colors)

    table = ax2.table(
        cellText=table_data,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        colWidths=[0.22, 0.15, 0.18, 0.22, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Bold header
    for j in range(5):
        table[0, j].set_text_props(fontweight='bold', fontsize=10)

    ax2.set_title("On-Demand Layering 전략 비교 요약", fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "03_strategy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 4: Per-stage VRAM for On-Demand Layering ────────────────────────

def create_stage_vram_chart():
    """Create per-stage VRAM visualization for on-demand layering."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: Sequential stage VRAM (BF16) ──
    ax1 = axes[0]

    stages = ["Vision\nEncoder", "VLM\nLM + Head", "Expert\n(Trajectory\nDecoder)"]
    vram_bf16 = [
        offload_data["per_stage_vram_gb"]["vision_encoder"],
        offload_data["per_stage_vram_gb"]["vlm_lm_plus_head"],
        offload_data["per_stage_vram_gb"]["expert"],
    ]

    # Estimate activation memory (~1-2 GB depending on stage)
    activation_est = [0.5, 2.0, 1.0]

    bars_weight = ax1.bar(stages, vram_bf16, color=[COLORS['vision'], COLORS['vlm_lm'], COLORS['expert']],
                          label='모델 가중치', edgecolor='white')
    bars_act = ax1.bar(stages, activation_est, bottom=vram_bf16,
                       color='#FFEB3B', alpha=0.7, label='활성화 메모리 (추정)', edgecolor='white')

    ax1.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(2.4, 12.3, "12 GB 제한", ha='right', fontsize=10, color='red', fontweight='bold')

    # Value labels
    for bar, val, act in zip(bars_weight, vram_bf16, activation_est):
        total = val + act
        ax1.text(bar.get_x() + bar.get_width()/2, total + 0.3,
                f'{val:.2f} + {act:.1f}\n= {total:.2f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel("VRAM 사용량 (GB)", fontsize=12)
    ax1.set_title("모듈별 순차 로딩 시 VRAM (BF16)", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 22)
    ax1.grid(axis='y', alpha=0.3)

    # ── Right: INT4 quantized comparison ──
    ax2 = axes[1]

    vram_int4 = [v / 4 for v in vram_bf16]  # Approximate INT4 compression
    activation_est_int4 = [0.5, 1.5, 0.8]  # Slightly less activation too

    bars_weight4 = ax2.bar(stages, vram_int4,
                           color=[COLORS['vision'], COLORS['vlm_lm'], COLORS['expert']],
                           alpha=0.8, label='모델 가중치 (INT4)', edgecolor='white')
    bars_act4 = ax2.bar(stages, activation_est_int4, bottom=vram_int4,
                        color='#FFEB3B', alpha=0.7, label='활성화 메모리 (추정)', edgecolor='white')

    ax2.axhline(y=12, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(2.4, 12.3, "12 GB 제한", ha='right', fontsize=10, color='red', fontweight='bold')

    for bar, val, act in zip(bars_weight4, vram_int4, activation_est_int4):
        total = val + act
        ax2.text(bar.get_x() + bar.get_width()/2, total + 0.15,
                f'{val:.2f} + {act:.1f}\n= {total:.2f} GB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel("VRAM 사용량 (GB)", fontsize=12)
    ax2.set_title("모듈별 순차 로딩 시 VRAM (INT4 양자화)", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.set_ylim(0, 8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "04_per_stage_vram.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Figure 5: On-Demand Layering Pipeline Diagram ──────────────────────────

def create_pipeline_diagram():
    """Create a visual diagram of the on-demand layering pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Timeline-based pipeline diagram
    # Show 3 phases with overlapping Read/Copy/Kernel

    # Phase labels
    phases = [
        {"name": "Phase 1: Vision Encoder", "start": 0, "end": 3, "color": COLORS['vision']},
        {"name": "Phase 2: VLM Language Model (36 layers)", "start": 3.5, "end": 16, "color": COLORS['vlm_lm']},
        {"name": "Phase 3: Expert Decoder (36 layers x 10 steps)", "start": 16.5, "end": 24, "color": COLORS['expert']},
    ]

    # Draw phase backgrounds
    for phase in phases:
        ax.axvspan(phase["start"], phase["end"], alpha=0.1, color=phase["color"])
        ax.text((phase["start"] + phase["end"]) / 2, 3.7, phase["name"],
               ha='center', va='center', fontsize=10, fontweight='bold',
               color=phase["color"],
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=phase["color"], alpha=0.9))

    # Pipeline rows
    row_labels = ["SSD/CPU Read", "CPU->GPU Copy", "GPU Kernel"]
    row_y = [2.5, 1.5, 0.5]

    # Draw pipeline blocks for VLM layers (show first few)
    for i in range(6):
        x_start = 4 + i * 1.5
        # Read block
        ax.add_patch(plt.Rectangle((x_start, 2.2), 1.2, 0.6, facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=1))
        ax.text(x_start + 0.6, 2.5, f"R{i}", ha='center', va='center', fontsize=8)
        # Copy block (shifted right)
        ax.add_patch(plt.Rectangle((x_start + 0.5, 1.2), 1.0, 0.6, facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=1))
        ax.text(x_start + 1.0, 1.5, f"C{i}", ha='center', va='center', fontsize=8)
        # Kernel block (shifted more right)
        ax.add_patch(plt.Rectangle((x_start + 1.0, 0.2), 1.2, 0.6, facecolor='#FFE0B2', edgecolor='#E65100', linewidth=1))
        ax.text(x_start + 1.6, 0.5, f"K{i}", ha='center', va='center', fontsize=8)

    # Add "..." for remaining layers
    ax.text(13.5, 2.5, "...", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(13.5, 1.5, "...", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(13.5, 0.5, "...", ha='center', va='center', fontsize=16, fontweight='bold')

    # Row labels
    for label, y in zip(row_labels, row_y):
        ax.text(-0.5, y, label, ha='right', va='center', fontsize=11, fontweight='bold')

    # VRAM usage annotation
    ax.annotate("VRAM: ~1.8 GB\n(더블 버퍼)",
                xy=(9, -0.3), fontsize=10,
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#4CAF50'))

    ax.annotate("VRAM: ~21.5 GB\n(전체 로딩)",
                xy=(20, -0.3), fontsize=10,
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#F44336'))

    # Add arrows showing pipeline overlap
    ax.annotate('', xy=(5.5, 2.8), xytext=(4, 2.8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(4.75, 3.0, "파이프라인 중첩", ha='center', fontsize=8, color='gray')

    ax.set_xlim(-2, 25)
    ax.set_ylim(-1, 4.5)
    ax.set_xlabel("시간 (상대)", fontsize=12)
    ax.set_title("On-Demand Layering 3단계 파이프라인: Read -> Copy -> Kernel", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#BBDEFB', edgecolor='#1565C0', label='Read (SSD/CPU -> CPU 메모리)'),
        Patch(facecolor='#C8E6C9', edgecolor='#2E7D32', label='Copy (CPU -> GPU 메모리)'),
        Patch(facecolor='#FFE0B2', edgecolor='#E65100', label='Kernel (GPU 연산)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "05_pipeline_diagram.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Part 3: Creating Visualizations")
    print("=" * 80)

    print("\n[1/5] Memory distribution chart...")
    create_memory_distribution()

    print("[2/5] device_map analysis chart...")
    create_device_map_analysis()

    print("[3/5] Strategy comparison chart...")
    create_comparison_chart()

    print("[4/5] Per-stage VRAM chart...")
    create_stage_vram_chart()

    print("[5/5] Pipeline diagram...")
    create_pipeline_diagram()

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
