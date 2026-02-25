#!/usr/bin/env python3
"""실험 6: 디퓨전 스텝 축소 + Expert 모델 경량화 (창의적 아이디어 1)

Alpamayo의 Flow Matching 디퓨전은 10 Euler 스텝으로 궤적을 생성.
이 스텝 수를 줄이면 Expert 모델 forward pass 횟수가 비례하여 감소.
또한 Expert 모듈과 VLM KV cache 공유 최적화를 탐색한다.
"""

import sys
import json
import gc
import time
import torch
import numpy as np

sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")


def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024**2


def analyze_diffusion_step_impact():
    """디퓨전 스텝 수 변경의 이론적 영향 분석."""

    results = {"diffusion_analysis": {}}

    # Alpamayo Flow Matching: 10 Euler steps
    # 각 스텝: Expert 모델 forward pass (action_in_proj → Expert Transformer → action_out_proj)
    # Expert는 text_config 기반이지만 더 작은 설정 가능

    # Expert 모델 구조 분석
    # config에서 expert_cfg로 설정되며, text_config를 기반으로 함
    # 기본: Qwen3 아키텍처 사용, 파라미터 수는 설정에 따라 다름

    # 디퓨전 스텝별 시간 예측
    baseline_steps = 10

    scenarios = []
    for steps in [1, 2, 3, 5, 8, 10, 15, 20]:
        speed_ratio = steps / baseline_steps
        # Flow Matching에서 스텝 수 줄이면 정확도 저하
        # 1 스텝: Consistency Model과 유사, 큰 오차
        # 5 스텝: 일반적으로 수용 가능
        # 10 스텝: 기본값 (충분한 품질)
        if steps <= 1:
            quality = "매우 낮음"
            quality_score = 2
        elif steps <= 3:
            quality = "낮음"
            quality_score = 5
        elif steps <= 5:
            quality = "중간"
            quality_score = 7
        elif steps <= 8:
            quality = "높음"
            quality_score = 9
        else:
            quality = "기본/최상"
            quality_score = 10

        scenarios.append({
            "num_steps": steps,
            "expert_forward_passes": steps,
            "speed_ratio_vs_baseline": round(speed_ratio, 2),
            "estimated_quality": quality,
            "quality_score": quality_score,
            "diffusion_time_proportion": round(speed_ratio, 2),
        })

    results["diffusion_analysis"]["step_scenarios"] = scenarios

    return results


def benchmark_diffusion_steps():
    """디퓨전 스텝 수에 따른 실제 연산 시간 벤치마크."""

    results = {"step_benchmark": []}

    torch.cuda.empty_cache()
    gc.collect()

    # Expert 모델 forward pass 시뮬레이션
    # Expert: Transformer decoder (텍스트 config 기반, 약 2.3B params)
    # action_in_proj: MLP (action_dim → hidden_size)
    # Expert: num_layers × (attention + FFN)
    # action_out_proj: Linear (hidden_size → action_dim)

    hidden_size = 4096  # Expert hidden size (text_config 기반)
    num_expert_layers = 28  # Expert는 보통 VLM보다 적은 레이어
    n_diffusion_tokens = 10  # action space tokens
    batch_size = 1
    dtype = torch.bfloat16

    # 간단한 Expert forward 시뮬레이션
    class SimpleExpertStep(torch.nn.Module):
        def __init__(self, h, n_layers, n_tokens):
            super().__init__()
            self.in_proj = torch.nn.Linear(7, h, bias=False)  # action_dim ≈ 7
            self.layers = torch.nn.ModuleList([
                torch.nn.TransformerEncoderLayer(h, 8, h * 4, batch_first=True, dtype=dtype)
                for _ in range(min(n_layers, 4))  # 4 레이어로 대표
            ])
            self.out_proj = torch.nn.Linear(h, 7, bias=False)

        def forward(self, x, t):
            x = self.in_proj(x)
            for layer in self.layers:
                x = layer(x)
            return self.out_proj(x)

    expert_sim = SimpleExpertStep(hidden_size, num_expert_layers, n_diffusion_tokens).to(dtype).to("cuda")

    for num_steps in [1, 2, 3, 5, 8, 10]:
        torch.cuda.empty_cache()

        # 랜덤 초기 noise
        x = torch.randn(batch_size, n_diffusion_tokens, 7, dtype=dtype, device="cuda")
        time_steps = torch.linspace(0, 1, num_steps + 1, device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                current = x.clone()
                for i in range(num_steps):
                    dt = time_steps[i+1] - time_steps[i]
                    t = time_steps[i].unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
                    v = expert_sim(current, t)
                    current = current + dt * v

        torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.perf_counter()

                current = x.clone()
                for i in range(num_steps):
                    dt = time_steps[i+1] - time_steps[i]
                    t = time_steps[i].unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
                    v = expert_sim(current, t)
                    current = current + dt * v

                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_time = float(np.mean(times)) * 1000
        std_time = float(np.std(times)) * 1000
        mem = get_gpu_memory_mb()

        results["step_benchmark"].append({
            "num_steps": num_steps,
            "avg_time_ms": round(avg_time, 3),
            "std_time_ms": round(std_time, 3),
            "gpu_memory_mb": round(mem, 1),
            "time_per_step_ms": round(avg_time / num_steps, 3),
        })

    del expert_sim, x
    torch.cuda.empty_cache()
    gc.collect()

    return results


def expert_model_optimization():
    """Expert 모델 경량화 옵션 분석."""

    results = {"expert_optimization": {}}

    # Expert 모델 현재 크기: ~2.3B params (text_config 기반)
    # 최적화 옵션들:

    strategies = [
        {
            "name": "Baseline Expert",
            "description": "현재 Expert 모델 (text_config 기반 Transformer)",
            "estimated_params": "~2.3B",
            "fp16_gb": 4.56,
            "int4_gb": 4.56 / 4,
            "quality_impact": "없음",
            "implementation": "현재 상태",
        },
        {
            "name": "Expert INT8 양자화",
            "description": "Expert Transformer를 INT8로 양자화",
            "estimated_params": "~2.3B",
            "fp16_gb": 4.56,
            "int4_gb": 4.56 / 2,
            "quality_impact": "경미",
            "implementation": "bitsandbytes INT8 또는 torch.ao.quantization",
        },
        {
            "name": "Expert 레이어 축소",
            "description": "Expert의 레이어 수를 50% 줄임 (distillation)",
            "estimated_params": "~1.15B",
            "fp16_gb": 4.56 / 2,
            "int4_gb": 4.56 / 8,
            "quality_impact": "중간 (distillation으로 완화 가능)",
            "implementation": "Knowledge distillation + layer pruning",
        },
        {
            "name": "Diffusion 5→3 스텝 + Expert INT8",
            "description": "디퓨전 스텝 축소 + Expert 양자화 조합",
            "estimated_params": "~2.3B",
            "fp16_gb": 4.56,
            "int4_gb": 4.56 / 2,
            "quality_impact": "중간",
            "implementation": "스텝 축소 + INT8 양자화",
            "speed_gain": "3x (스텝 축소) + 약간 (양자화)",
        },
        {
            "name": "KV cache 재사용 최적화",
            "description": "VLM의 KV cache를 Expert에서 효율적으로 재사용",
            "estimated_params": "~2.3B",
            "fp16_gb": 4.56,
            "int4_gb": None,
            "quality_impact": "없음",
            "implementation": "crop/reuse 패턴 최적화 (이미 구현됨, prompt_cache.crop())",
        },
    ]

    results["expert_optimization"]["strategies"] = strategies

    # 전체 파이프라인 시간 분석
    # Baseline: Vision(~1%) + VLM Generate(~60%) + Expert/Diffusion(~39%)
    # (4-bit 기준: 총 4.91초 중 추정)
    results["expert_optimization"]["pipeline_time_breakdown"] = {
        "total_4bit_seconds": 4.91,
        "estimated_vision_pct": 2,
        "estimated_vlm_generate_pct": 60,
        "estimated_expert_diffusion_pct": 38,
        "estimated_expert_per_step_ms": 4.91 * 0.38 / 10 * 1000,  # ~186.6 ms/step
    }

    return results


def consistency_distillation_analysis():
    """Consistency Distillation 가능성 분석.
    Flow Matching 모델을 1-step 모델로 distillation하는 방법."""

    return {
        "consistency_distillation": {
            "description": "Flow Matching → Consistency Model 변환",
            "principle": "다수 스텝의 ODE 궤적을 1-2 스텝으로 학습 (Consistency Models, Song et al. 2023)",
            "applicability_to_alpamayo": {
                "score": "중간-높음",
                "reason": "Alpamayo의 10-step Euler 적분을 1-2 step으로 줄일 수 있음. "
                          "단, 궤적 예측 정확도 검증 필요.",
            },
            "implementation_steps": [
                "1. Teacher: 현재 10-step Flow Matching 모델",
                "2. Student: 동일 구조의 1-2 step 모델",
                "3. Teacher의 ODE 궤적을 따라 Student 학습",
                "4. Self-consistency loss 적용",
            ],
            "expected_gain": {
                "speed": "5-10x (디퓨전 단계만)",
                "memory": "동일 (모델 크기는 같음)",
                "quality": "약간 저하 (1-step) ~ 거의 동일 (2-step)",
            },
            "challenges": [
                "학습 데이터/환경 필요 (커스텀 학습 파이프라인)",
                "Alpamayo의 action space 특성에 맞는 loss 설계",
                "VLM KV cache와의 상호작용 고려",
            ],
            "difficulty": "어려움 (학습 필요)",
        }
    }


def main():
    print("=" * 70)
    print("실험 6: 디퓨전 스텝 축소 + Expert 모델 경량화")
    print("=" * 70)

    all_results = {}

    # 1. 디퓨전 스텝 영향 분석
    print("\n[1/4] 디퓨전 스텝 수 영향 분석...")
    step_analysis = analyze_diffusion_step_impact()
    all_results.update(step_analysis)

    print(f"\n  {'스텝 수':>8} {'속도 비율':>10} {'품질':>10} {'점수':>6}")
    print("  " + "-" * 40)
    for s in step_analysis["diffusion_analysis"]["step_scenarios"]:
        print(f"  {s['num_steps']:>8} {s['speed_ratio_vs_baseline']:>8.2f}x {s['estimated_quality']:>10} {s['quality_score']:>4}/10")

    # 2. 디퓨전 스텝 벤치마크
    print("\n[2/4] 디퓨전 스텝 벤치마크...")
    bench_results = benchmark_diffusion_steps()
    all_results.update(bench_results)

    baseline_time = None
    print(f"\n  {'스텝':>6} {'총 시간':>12} {'스텝당 시간':>12} {'Speedup':>10}")
    print("  " + "-" * 45)
    for b in bench_results["step_benchmark"]:
        if baseline_time is None and b["num_steps"] == 10:
            baseline_time = b["avg_time_ms"]
        speedup = baseline_time / b["avg_time_ms"] if baseline_time and b["avg_time_ms"] > 0 else 0
        print(f"  {b['num_steps']:>6} {b['avg_time_ms']:>10.3f}ms {b['time_per_step_ms']:>10.3f}ms {speedup:>8.2f}x")

    # 3. Expert 모델 최적화
    print("\n[3/4] Expert 모델 최적화 분석...")
    expert_results = expert_model_optimization()
    all_results.update(expert_results)

    for s in expert_results["expert_optimization"]["strategies"]:
        print(f"  {s['name']}: {s['description']}")
        print(f"    FP16: {s['fp16_gb']:.2f}GB, INT4: {s['int4_gb']:.2f}GB" if s['int4_gb'] else f"    FP16: {s['fp16_gb']:.2f}GB")
        print(f"    품질 영향: {s['quality_impact']}")

    # 4. Consistency Distillation 분석
    print("\n[4/4] Consistency Distillation 가능성 분석...")
    cd_results = consistency_distillation_analysis()
    all_results.update(cd_results)

    cd = cd_results["consistency_distillation"]
    print(f"  적용 가능성: {cd['applicability_to_alpamayo']['score']}")
    print(f"  이유: {cd['applicability_to_alpamayo']['reason']}")
    print(f"  예상 속도 향상: {cd['expected_gain']['speed']}")

    # 결과 저장
    output_path = "/home/seungwoo/workspace/research/04-creative-approaches/exp06_diffusion_step_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)

    print(f"\n결과 저장: {output_path}")
    print("실험 6 완료!")


if __name__ == "__main__":
    main()
