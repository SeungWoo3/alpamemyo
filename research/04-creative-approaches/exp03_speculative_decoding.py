#!/usr/bin/env python3
"""실험 3: Speculative Decoding 가능성 분석

Alpamayo의 VLM 생성 과정에서 speculative decoding 적용 가능성을 분석한다.
"""

import sys
import json
import gc
import time
import torch
import numpy as np

sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")

def analyze_speculative_decoding_feasibility():
    """Speculative decoding 적용 가능성 이론 분석."""

    analysis = {
        "alpamayo_pipeline": {
            "description": "Alpamayo 추론 파이프라인 분석",
            "stages": [
                {
                    "stage": "1. Vision Encoding",
                    "type": "parallel (batch)",
                    "speculative_applicable": False,
                    "reason": "비자기회귀적. 전체 이미지를 한번에 처리."
                },
                {
                    "stage": "2. VLM Autoregressive Generation (CoT)",
                    "type": "autoregressive",
                    "speculative_applicable": True,
                    "reason": "토큰 단위 자기회귀 생성. 최대 256 토큰. Speculative decoding 적용 가능.",
                    "max_tokens": 256,
                },
                {
                    "stage": "3. Expert + Diffusion (Trajectory)",
                    "type": "diffusion (10 Euler steps)",
                    "speculative_applicable": False,
                    "reason": "Flow Matching 기반 디퓨전. 고정 10스텝. Speculative decoding 비해당."
                }
            ]
        }
    }

    # VLM 생성 단계만 speculative decoding 대상
    # Qwen3-VL-8B → draft model로 Qwen3-VL-2B 사용 가능성

    analysis["draft_model_candidates"] = {
        "option_1": {
            "name": "Qwen3-VL-2B-Instruct (네이티브 draft)",
            "params": "2B",
            "fp16_size_gb": 4.0,
            "int4_size_gb": 1.0,
            "vocab_compatible": True,
            "architecture_compatible": True,
            "feasibility": "높음",
            "notes": "같은 Qwen3-VL 계열이라 vocab과 구조 호환. 하지만 Alpamayo의 커스텀 토큰(traj_token 등) 처리 필요.",
        },
        "option_2": {
            "name": "Alpamayo-INT4 (self-speculative)",
            "params": "10B (INT4)",
            "fp16_size_gb": None,
            "int4_size_gb": 5.23,
            "vocab_compatible": True,
            "architecture_compatible": True,
            "feasibility": "중간",
            "notes": "같은 모델의 INT4 버전을 draft로 사용. Self-speculative decoding. 메모리 이슈로 두 모델 동시 로드 어려움.",
        },
        "option_3": {
            "name": "Layer-skipping draft (자체 모델 일부 레이어)",
            "params": "~2-4B (9-18 layers)",
            "fp16_size_gb": None,
            "int4_size_gb": None,
            "vocab_compatible": True,
            "architecture_compatible": True,
            "feasibility": "중간-높음",
            "notes": "Alpamayo VLM에서 일부 레이어만 사용하여 draft 생성. 추가 메모리 불필요. 'LayerSkip' 기법.",
        },
    }

    return analysis


def benchmark_draft_model_speed():
    """Draft 모델 속도 시뮬레이션: 작은 모델 vs 큰 모델 forward pass 비교."""

    results = {"speed_comparison": {}}

    torch.cuda.empty_cache()
    gc.collect()

    # 시뮬레이션: Transformer 디코더 레이어의 forward pass 시간
    batch_size = 1
    seq_len = 1  # 자기회귀 디코딩 시 1 토큰씩
    hidden_size_large = 4096  # Qwen3-VL-8B
    hidden_size_small = 1536  # Qwen3-VL-2B (추정)

    for name, (h_size, n_layers, kv_len) in [
        ("large_8B_full", (hidden_size_large, 36, 1024)),
        ("large_8B_half_layers", (hidden_size_large, 18, 1024)),
        ("large_8B_quarter_layers", (hidden_size_large, 9, 1024)),
        ("small_2B", (hidden_size_small, 28, 1024)),
    ]:
        torch.cuda.empty_cache()

        # 간단한 Transformer-like 연산 시뮬레이션
        x = torch.randn(batch_size, seq_len, h_size, dtype=torch.bfloat16, device="cuda")

        # Attention + FFN 시뮬레이션
        attn_weight = torch.randn(h_size, h_size, dtype=torch.bfloat16, device="cuda")
        ffn_weight1 = torch.randn(h_size * 4, h_size, dtype=torch.bfloat16, device="cuda")
        ffn_weight2 = torch.randn(h_size, h_size * 4, dtype=torch.bfloat16, device="cuda")

        # Warmup
        for _ in range(5):
            out = x
            for _ in range(min(n_layers, 3)):
                out = out @ attn_weight.T
                ffn_out = out @ ffn_weight1.T
                out = ffn_out @ ffn_weight2.T

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(10):
            out = x
            for _ in range(n_layers):
                out = out @ attn_weight.T
                ffn_out = out @ ffn_weight1.T
                out = ffn_out @ ffn_weight2.T

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 10

        mem_mb = get_gpu_memory_mb()

        results["speed_comparison"][name] = {
            "hidden_size": h_size,
            "num_layers": n_layers,
            "forward_time_ms": round(elapsed * 1000, 3),
            "gpu_memory_mb": round(mem_mb, 1),
        }

        del x, attn_weight, ffn_weight1, ffn_weight2
        torch.cuda.empty_cache()

    return results


def get_gpu_memory_mb():
    return torch.cuda.memory_allocated() / 1024**2


def speculative_decoding_gain_model():
    """Speculative decoding 이론적 속도 향상 모델."""

    results = {"theoretical_gain": {}}

    # 파라미터 설정
    # T_target: target 모델 1 토큰 생성 시간 (ms)
    # T_draft: draft 모델 1 토큰 생성 시간 (ms)
    # alpha: acceptance rate (draft 토큰이 target에 의해 수용되는 비율)
    # gamma: speculative 윈도우 크기 (한번에 draft가 생성하는 토큰 수)

    # Alpamayo 추정 시간
    # FP16 VLM: 전체 추론 273.79초 중 VLM 생성이 대부분
    # INT4 VLM: 전체 추론 4.91초
    # 생성 토큰 수: 약 200 (CoT ~150 + meta_action ~30 + etc)
    est_tokens = 200
    t_target_fp16 = 273.79 * 1000 / est_tokens  # ms/token (대략적)
    t_target_int4 = 4.91 * 1000 / est_tokens  # ms/token

    scenarios = []

    for gamma in [2, 4, 6, 8]:
        for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
            # 기대 수용 토큰 수: sum_{i=0}^{gamma} alpha^i = (1 - alpha^(gamma+1)) / (1 - alpha)
            expected_accepted = (1 - alpha**(gamma + 1)) / (1 - alpha)

            # Speculative decoding 시간:
            # draft: gamma * T_draft
            # verify: 1 * T_target (gamma+1 토큰을 병렬 verify)
            # 생성 토큰: expected_accepted

            for draft_name, t_draft, t_target in [
                ("2B_draft_for_FP16", 5.0, t_target_fp16),
                ("layer_skip_for_FP16", 200.0, t_target_fp16),
                ("2B_draft_for_INT4", 5.0, t_target_int4),
                ("layer_skip_for_INT4", 5.0, t_target_int4),
            ]:
                # Wall-clock time per speculative iteration
                spec_time = gamma * t_draft + t_target
                # Tokens per iteration
                tokens_per_iter = expected_accepted
                # Time per token (speculative)
                time_per_token_spec = spec_time / tokens_per_iter if tokens_per_iter > 0 else float("inf")
                # Speedup
                speedup = t_target / time_per_token_spec if time_per_token_spec > 0 else 0

                scenarios.append({
                    "draft_model": draft_name,
                    "gamma": gamma,
                    "alpha": alpha,
                    "t_draft_ms": t_draft,
                    "t_target_ms": round(t_target, 1),
                    "expected_accepted_tokens": round(expected_accepted, 2),
                    "time_per_token_ms": round(time_per_token_spec, 2),
                    "speedup": round(speedup, 2),
                })

    results["theoretical_gain"] = scenarios
    return results


def alpamayo_specific_challenges():
    """Alpamayo 모델에서 speculative decoding 적용 시 고유 도전과제."""

    return {
        "challenges": [
            {
                "issue": "커스텀 토큰 (trajectory tokens)",
                "description": "Alpamayo는 768개의 trajectory 이산 토큰(<i0>~<i767>)과 특수 제어 토큰을 사용. Draft 모델도 이 토큰을 알아야 함.",
                "severity": "높음",
                "solution": "Draft 모델에도 같은 토큰을 추가하고 fine-tuning 필요. 또는 layer-skipping 방식으로 같은 vocab 공유.",
            },
            {
                "issue": "ExpertLogitsProcessor",
                "description": "VLM 생성 시 trajectory 토큰을 마스킹하는 LogitsProcessor가 적용됨. Draft 모델에도 동일하게 적용해야.",
                "severity": "중간",
                "solution": "Draft 모델에도 동일한 LogitsProcessor 적용.",
            },
            {
                "issue": "KV cache 공유 (Expert 모델)",
                "description": "VLM의 KV cache가 Expert 모델의 입력으로 사용됨 (past_key_values). Speculative decoding 시 KV cache 관리가 복잡해짐.",
                "severity": "높음",
                "solution": "Speculative decoding을 VLM 단계에만 적용하고, Expert 단계는 정상 실행.",
            },
            {
                "issue": "StopAfterEOS 커스텀 종료 조건",
                "description": "<traj_future_start> 토큰 생성 후 1토큰 더 생성하는 커스텀 종료 조건. Draft 모델의 종료 타이밍이 달라질 수 있음.",
                "severity": "중간",
                "solution": "Draft 모델에서도 동일한 StoppingCriteria 사용.",
            },
            {
                "issue": "12GB VRAM 제약",
                "description": "Target + Draft 모델을 동시에 GPU에 로드하려면 메모리 제약이 심각. INT4 target(~5.2GB) + INT4 2B draft(~1GB) = ~6.2GB + KV cache + activations.",
                "severity": "높음",
                "solution": "Layer-skipping 방식이 추가 메모리 없이 사용 가능. 또는 draft를 CPU에서 실행.",
            },
        ],
        "overall_feasibility": {
            "layer_skipping": "중간-높음 (추가 메모리 불필요, 같은 vocab, 구현 복잡도 보통)",
            "separate_draft_model": "낮음-중간 (메모리 문제, vocab 매칭 필요, 커스텀 토큰 fine-tuning)",
            "int4_self_speculative": "낮음 (두 모델 동시 로드 불가능, 같은 구조라 속도 차이 미미)",
        }
    }


def main():
    print("=" * 70)
    print("실험 3: Speculative Decoding 가능성 분석")
    print("=" * 70)

    all_results = {}

    # 1. 적용 가능성 분석
    print("\n[1/4] Speculative Decoding 적용 가능성 분석...")
    feasibility = analyze_speculative_decoding_feasibility()
    all_results.update(feasibility)

    for stage in feasibility["alpamayo_pipeline"]["stages"]:
        applicable = "적용 가능" if stage["speculative_applicable"] else "비해당"
        print(f"  {stage['stage']}: {applicable}")
        print(f"    -> {stage['reason']}")

    print("\n  Draft 모델 후보:")
    for key, info in feasibility["draft_model_candidates"].items():
        print(f"  {info['name']}: {info['feasibility']} ({info['notes'][:60]}...)")

    # 2. Draft vs Target 속도 비교
    print("\n[2/4] Draft vs Target 모델 속도 비교...")
    speed_results = benchmark_draft_model_speed()
    all_results.update(speed_results)

    print(f"\n  {'모델':<30} {'Layers':>8} {'Time (ms)':>12} {'VRAM (MB)':>12}")
    print("  " + "-" * 65)
    for name, data in speed_results["speed_comparison"].items():
        print(f"  {name:<30} {data['num_layers']:>8} {data['forward_time_ms']:>10.3f} ms {data['gpu_memory_mb']:>10.1f} MB")

    # 3. 이론적 속도 향상 모델
    print("\n[3/4] Speculative Decoding 이론적 속도 향상 모델...")
    gain_results = speculative_decoding_gain_model()
    all_results.update(gain_results)

    # 최적 결과만 출력
    best_by_draft = {}
    for s in gain_results["theoretical_gain"]:
        key = s["draft_model"]
        if key not in best_by_draft or s["speedup"] > best_by_draft[key]["speedup"]:
            best_by_draft[key] = s

    print(f"\n  {'Draft Model':<30} {'gamma':>6} {'alpha':>6} {'Speedup':>10}")
    print("  " + "-" * 55)
    for key, s in best_by_draft.items():
        print(f"  {key:<30} {s['gamma']:>6} {s['alpha']:>6.1f} {s['speedup']:>8.2f}x")

    # 4. Alpamayo 고유 도전과제
    print("\n[4/4] Alpamayo 고유 도전과제 분석...")
    challenges = alpamayo_specific_challenges()
    all_results.update(challenges)

    for c in challenges["challenges"]:
        print(f"  [{c['severity']}] {c['issue']}: {c['description'][:80]}...")

    print(f"\n  종합 평가:")
    for method, assessment in challenges["overall_feasibility"].items():
        print(f"    {method}: {assessment}")

    # 결과 저장
    output_path = "/home/seungwoo/workspace/research/04-creative-approaches/exp03_speculative_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)

    print(f"\n결과 저장: {output_path}")
    print("실험 3 완료!")


if __name__ == "__main__":
    main()
