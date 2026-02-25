#!/usr/bin/env python3
"""시각화: 연구 방향 4 실험 결과 그래프 생성."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한국어 폰트 설정
font_path = "/home/seungwoo/.local/share/fonts/NanumGothic-Regular.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

FIGURES_DIR = "/home/seungwoo/workspace/research/04-creative-approaches/figures"


def fig01_hybrid_quantization_memory():
    """그림 1: 하이브리드 양자화 시나리오별 메모리 비교."""
    with open("/home/seungwoo/workspace/research/04-creative-approaches/exp01_hybrid_quant_results.json") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    names = []
    sizes = []
    for key in ["baseline_fp16", "full_int8", "attn_int4_ffn_int8", "attn_int8_ffn_int4", "full_int4"]:
        s = scenarios[key]
        names.append(s["description"])
        sizes.append(s["total_gb"])

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax.barh(names, sizes, color=colors, edgecolor="black", linewidth=0.5)

    # 12GB 한계선
    ax.axvline(x=12.0, color="red", linestyle="--", linewidth=2, label="12GB VRAM 한계")

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{size:.2f} GB", va="center", fontsize=11, fontproperties=font_prop)

    ax.set_xlabel("VLM 가중치 크기 (GB)", fontsize=12, fontproperties=font_prop)
    ax.set_title("하이브리드 양자화 시나리오별 VLM 메모리 비교", fontsize=14, fontproperties=font_prop, fontweight="bold")
    ax.legend(prop=font_prop, fontsize=11)
    ax.set_xlim(0, 25)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_hybrid_quantization_memory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  01_hybrid_quantization_memory.png 생성 완료")


def fig02_kv_cache_growth():
    """그림 2: KV Cache 크기 증가 (토큰 수 기준)."""
    with open("/home/seungwoo/workspace/research/04-creative-approaches/exp02_kv_cache_results.json") as f:
        data = json.load(f)

    sim = data["kv_cache_simulation"]
    tokens = [s["tokens"] for s in sim]
    actual = [s["actual_kv_mb"] for s in sim]
    theory = [s["theory_kv_mb"] for s in sim]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 왼쪽: KV cache 크기
    ax1.plot(tokens, actual, "bo-", linewidth=2, markersize=8, label="실측값")
    ax1.plot(tokens, theory, "r--", linewidth=2, label="이론값")
    ax1.axhline(y=2235, color="orange", linestyle=":", linewidth=2, label="Alpamayo 추정 (2.18 GB)")
    ax1.set_xlabel("토큰 수", fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel("KV Cache 크기 (MB)", fontsize=12, fontproperties=font_prop)
    ax1.set_title("토큰 수에 따른 KV Cache 크기", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax1.legend(prop=font_prop, fontsize=10)
    ax1.grid(True, alpha=0.3)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 오른쪽: 압축 기법 비교
    comp = data["compression_methods"]
    methods = ["원본 (BF16)", "INT8 양자화", "INT4 양자화\n(group=32)", "Token 50%\nEviction", "Token 25%\nEviction"]
    savings = [0, 48.4, 68.8, 50.0, 75.0]
    errors = [0, 2.92, 25.79, None, None]  # relative error %

    colors = ["#3498db", "#2ecc71", "#27ae60", "#e67e22", "#e74c3c"]
    bars = ax2.bar(methods, savings, color=colors, edgecolor="black", linewidth=0.5)

    for bar, s in zip(bars, savings):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{s:.1f}%", ha="center", fontsize=10, fontproperties=font_prop)

    ax2.set_ylabel("메모리 절감률 (%)", fontsize=12, fontproperties=font_prop)
    ax2.set_title("KV Cache 압축 기법별 절감률", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax2.set_ylim(0, 90)
    ax2.grid(True, alpha=0.3, axis="y")
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_kv_cache_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  02_kv_cache_analysis.png 생성 완료")


def fig03_pruning_scenarios():
    """그림 3: 레이어 가지치기 시나리오 메모리/품질 트레이드오프."""
    with open("/home/seungwoo/workspace/research/04-creative-approaches/exp04_layer_pruning_results.json") as f:
        data = json.load(f)

    scenarios = data["pruning_scenarios"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    layers_kept = [s["layers_kept"] for s in scenarios]
    vlm_fp16 = [s["vlm_fp16_gb"] for s in scenarios]
    vlm_int4 = [s["vlm_int4_gb"] for s in scenarios]
    quality_labels = [s["expected_quality_impact"] for s in scenarios]

    x = np.arange(len(layers_kept))
    width = 0.35

    bars1 = ax1.bar(x - width/2, vlm_fp16, width, label="FP16", color="#e74c3c", edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + width/2, vlm_int4, width, label="INT4", color="#27ae60", edgecolor="black", linewidth=0.5)
    ax1.axhline(y=12, color="red", linestyle="--", linewidth=2, alpha=0.7, label="12GB VRAM 한계")
    ax1.set_xlabel("유지 레이어 수 (36개 중)", fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel("VLM 크기 (GB)", fontsize=12, fontproperties=font_prop)
    ax1.set_title("레이어 가지치기에 따른 메모리 절감", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{k}개\n({quality_labels[i]})" for i, k in enumerate(layers_kept)],
                        fontproperties=font_prop, fontsize=9)
    ax1.legend(prop=font_prop, fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 오른쪽: 전체 모델 크기 (VLM INT4 + 나머지 FP16)
    total_vlm_int4 = [s["total_vlm_int4_rest_fp16_gb"] for s in scenarios]
    colors = ["#27ae60" if t <= 12 else "#e74c3c" for t in total_vlm_int4]
    bars = ax2.bar(x, total_vlm_int4, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=12, color="red", linestyle="--", linewidth=2, alpha=0.7, label="12GB VRAM 한계")

    for bar, t in zip(bars, total_vlm_int4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{t:.1f}GB", ha="center", fontsize=10, fontproperties=font_prop)

    ax2.set_xlabel("유지 레이어 수", fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel("전체 모델 크기 (GB)", fontsize=12, fontproperties=font_prop)
    ax2.set_title("VLM-INT4 + 나머지 FP16 전체 크기", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in layers_kept], fontproperties=font_prop)
    ax2.legend(prop=font_prop, fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/03_layer_pruning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  03_layer_pruning.png 생성 완료")


def fig04_dynamic_resolution():
    """그림 4: 동적 해상도 트레이드오프."""
    with open("/home/seungwoo/workspace/research/04-creative-approaches/exp05_dynamic_resolution_results.json") as f:
        data = json.load(f)

    trade_off = data["trade_off_analysis"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [s["name"].split("(")[0].strip() for s in trade_off]
    mem_savings = [s["memory_savings_pct"] for s in trade_off]
    speed_improve = [s["speed_improvement_pct"] for s in trade_off]
    quality = [s["quality_score"] for s in trade_off]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mem_savings, width, label="메모리 절감 (%)", color="#3498db", edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + width/2, speed_improve, width, label="속도 향상 (%)", color="#e67e22", edgecolor="black", linewidth=0.5)

    ax1.set_xlabel("해상도 설정", fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel("개선율 (%)", fontsize=12, fontproperties=font_prop)
    ax1.set_title("해상도 변경에 따른 메모리/속도 개선", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontproperties=font_prop, fontsize=9, rotation=15)
    ax1.legend(prop=font_prop, fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 오른쪽: 품질 점수
    colors = ["#27ae60" if q >= 7 else "#e67e22" if q >= 5 else "#e74c3c" for q in quality]
    bars = ax2.bar(names, quality, color=colors, edgecolor="black", linewidth=0.5)

    for bar, q in zip(bars, quality):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{q}/10", ha="center", fontsize=11, fontproperties=font_prop)

    ax2.set_xlabel("해상도 설정", fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel("예상 품질 점수", fontsize=12, fontproperties=font_prop)
    ax2.set_title("해상도 변경에 따른 품질 영향", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax2.set_ylim(0, 12)
    ax2.set_xticklabels(names, fontproperties=font_prop, fontsize=9, rotation=15)
    ax2.grid(True, alpha=0.3, axis="y")
    for label in ax2.get_yticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_dynamic_resolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  04_dynamic_resolution.png 생성 완료")


def fig05_diffusion_steps():
    """그림 5: 디퓨전 스텝 축소 효과."""
    with open("/home/seungwoo/workspace/research/04-creative-approaches/exp06_diffusion_step_results.json") as f:
        data = json.load(f)

    bench = data["step_benchmark"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    steps = [b["num_steps"] for b in bench]
    times = [b["avg_time_ms"] for b in bench]
    per_step = [b["time_per_step_ms"] for b in bench]

    # 스텝 수에 따른 시간
    ax1.plot(steps, times, "bo-", linewidth=2, markersize=10)
    for s, t in zip(steps, times):
        ax1.annotate(f"{t:.1f}ms", (s, t), textcoords="offset points",
                     xytext=(10, 5), fontsize=10, fontproperties=font_prop)

    ax1.set_xlabel("디퓨전 스텝 수", fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel("총 시간 (ms)", fontsize=12, fontproperties=font_prop)
    ax1.set_title("디퓨전 스텝 수에 따른 추론 시간", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 품질-속도 트레이드오프
    theory = data["diffusion_analysis"]["step_scenarios"]
    th_steps = [s["num_steps"] for s in theory if s["num_steps"] <= 10]
    th_quality = [s["quality_score"] for s in theory if s["num_steps"] <= 10]
    th_speed = [s["speed_ratio_vs_baseline"] for s in theory if s["num_steps"] <= 10]

    scatter = ax2.scatter(th_speed, th_quality, c=th_steps, cmap="RdYlGn_r",
                          s=200, edgecolors="black", linewidth=1, zorder=5)
    for s, sp, q in zip(th_steps, th_speed, th_quality):
        ax2.annotate(f"{s}스텝", (sp, q), textcoords="offset points",
                     xytext=(10, 5), fontsize=10, fontproperties=font_prop)

    ax2.set_xlabel("속도 비율 (baseline 대비)", fontsize=12, fontproperties=font_prop)
    ax2.set_ylabel("예상 품질 점수 (10점 만점)", fontsize=12, fontproperties=font_prop)
    ax2.set_title("디퓨전 스텝: 속도 vs 품질", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax2.set_ylim(0, 11)
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("스텝 수", fontproperties=font_prop, fontsize=11)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/05_diffusion_steps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  05_diffusion_steps.png 생성 완료")


def fig06_comprehensive_priority():
    """그림 6: 종합 우선순위 매트릭스."""

    ideas = [
        "하이브리드\n양자화",
        "KV Cache\n압축",
        "Speculative\nDecoding",
        "레이어\n가지치기",
        "동적 해상도\n조절",
        "디퓨전 스텝\n축소",
        "Token Merging\n+ Chunked Prefill",
    ]

    # 가능성 (1-5), 효과 (1-5), 구현 용이성 (1-5, 높을수록 쉬움)
    feasibility = [5, 4, 2, 3, 4, 5, 4]
    impact = [5, 3, 4, 4, 3, 3, 3]
    ease = [3, 3, 1, 2, 5, 4, 3]

    # 종합 점수 = feasibility × impact × ease / 25 (정규화)
    composite = [f * i * e / 25 for f, i, e in zip(feasibility, impact, ease)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 왼쪽: 3차원 비교 (bar chart)
    x = np.arange(len(ideas))
    width = 0.25

    b1 = ax1.bar(x - width, feasibility, width, label="실현 가능성", color="#3498db", edgecolor="black", linewidth=0.5)
    b2 = ax1.bar(x, impact, width, label="예상 효과", color="#e67e22", edgecolor="black", linewidth=0.5)
    b3 = ax1.bar(x + width, ease, width, label="구현 용이성", color="#27ae60", edgecolor="black", linewidth=0.5)

    ax1.set_xlabel("연구 아이디어", fontsize=12, fontproperties=font_prop)
    ax1.set_ylabel("점수 (5점 만점)", fontsize=12, fontproperties=font_prop)
    ax1.set_title("아이디어별 평가 (가능성/효과/용이성)", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ideas, fontproperties=font_prop, fontsize=9)
    ax1.legend(prop=font_prop, fontsize=10, loc="upper right")
    ax1.set_ylim(0, 6)
    ax1.grid(True, alpha=0.3, axis="y")
    for label in ax1.get_yticklabels():
        label.set_fontproperties(font_prop)

    # 오른쪽: 종합 점수 순위
    sorted_idx = np.argsort(composite)[::-1]
    sorted_names = [ideas[i] for i in sorted_idx]
    sorted_scores = [composite[i] for i in sorted_idx]
    colors = ["#f39c12" if i < 3 else "#95a5a6" for i in range(len(sorted_scores))]

    bars = ax2.barh(range(len(sorted_names)), sorted_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels(sorted_names, fontproperties=font_prop, fontsize=10)
    ax2.invert_yaxis()

    for bar, score in zip(bars, sorted_scores):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f"{score:.2f}", va="center", fontsize=11, fontproperties=font_prop, fontweight="bold")

    ax2.set_xlabel("종합 점수 (가능성 x 효과 x 용이성 / 25)", fontsize=11, fontproperties=font_prop)
    ax2.set_title("종합 우선순위 (Top 3 = 금색)", fontsize=13, fontproperties=font_prop, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/06_priority_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  06_priority_matrix.png 생성 완료")


def fig07_offloading_strategies():
    """그림 7: 모듈별 오프로딩 전략 Peak VRAM 비교."""
    with open("/home/seungwoo/workspace/research/04-creative-approaches/exp01_hybrid_quant_results.json") as f:
        data = json.load(f)

    strategies = data["offloading_strategies"]

    fig, ax = plt.subplots(figsize=(12, 6))

    names = [s["name"] for s in strategies]
    vrams = [s["peak_vram_gb"] for s in strategies]
    colors = ["#27ae60" if v <= 10 else "#e67e22" if v <= 12 else "#e74c3c" for v in vrams]

    bars = ax.barh(names, vrams, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=12.0, color="red", linestyle="--", linewidth=2, label="12GB VRAM 한계")
    ax.axvline(x=10.0, color="orange", linestyle=":", linewidth=1.5, label="10GB 안전 마진")

    for bar, v in zip(bars, vrams):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{v:.2f} GB", va="center", fontsize=11, fontproperties=font_prop)

    ax.set_xlabel("Peak VRAM (GB)", fontsize=12, fontproperties=font_prop)
    ax.set_title("모듈별 양자화 + 오프로딩 전략의 Peak VRAM", fontsize=14, fontproperties=font_prop, fontweight="bold")
    ax.legend(prop=font_prop, fontsize=11)
    ax.set_xlim(0, 15)

    for label in ax.get_yticklabels() + ax.get_xticklabels():
        label.set_fontproperties(font_prop)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/07_offloading_strategies.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  07_offloading_strategies.png 생성 완료")


def main():
    print("시각화 생성 시작...")
    fig01_hybrid_quantization_memory()
    fig02_kv_cache_growth()
    fig03_pruning_scenarios()
    fig04_dynamic_resolution()
    fig05_diffusion_steps()
    fig06_comprehensive_priority()
    fig07_offloading_strategies()
    print("\n모든 시각화 생성 완료!")


if __name__ == "__main__":
    main()
