#!/usr/bin/env python3
"""실험 1: 하이브리드 양자화 (Mixed Precision Offloading) 분석

VLM의 attention/FFN 레이어별 다른 양자화 적용 시 메모리 효과 분석.
실제 모델 가중치를 로드하여 레이어별 크기/양자화 가능성을 평가한다.
"""

import sys
import json
import gc
import time
import torch
import numpy as np

sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")

def get_gpu_memory_mb():
    """현재 GPU 메모리 사용량을 MB 단위로 반환."""
    return torch.cuda.memory_allocated() / 1024**2

def get_gpu_reserved_mb():
    """현재 GPU 예약 메모리를 MB 단위로 반환."""
    return torch.cuda.memory_reserved() / 1024**2

def analyze_model_structure():
    """모델 구조를 분석하여 각 레이어 크기와 양자화 적용 가능성을 평가."""
    from transformers import Qwen3VLConfig

    # 먼저 config만 로드하여 구조 분석
    config = Qwen3VLConfig.from_pretrained("nvidia/Alpamayo-R1-10B")
    text_config = config.text_config

    results = {
        "model_info": {
            "hidden_size": text_config.hidden_size,
            "intermediate_size": text_config.intermediate_size,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": text_config.num_key_value_heads,
            "num_hidden_layers": text_config.num_hidden_layers,
            "head_dim": getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads),
            "vocab_size": text_config.vocab_size,
        }
    }

    hidden = text_config.hidden_size
    intermediate = text_config.intermediate_size
    n_heads = text_config.num_attention_heads
    n_kv_heads = text_config.num_key_value_heads
    head_dim = results["model_info"]["head_dim"]
    n_layers = text_config.num_hidden_layers

    # Attention 레이어 파라미터 수 계산
    # Q: hidden_size -> num_heads * head_dim
    # K: hidden_size -> num_kv_heads * head_dim
    # V: hidden_size -> num_kv_heads * head_dim
    # O: num_heads * head_dim -> hidden_size
    q_params = hidden * n_heads * head_dim
    k_params = hidden * n_kv_heads * head_dim
    v_params = hidden * n_kv_heads * head_dim
    o_params = n_heads * head_dim * hidden
    attn_params_per_layer = q_params + k_params + v_params + o_params

    # FFN (MLP) 레이어 파라미터 수 (Qwen uses SwiGLU: gate_proj, up_proj, down_proj)
    gate_params = hidden * intermediate
    up_params = hidden * intermediate
    down_params = intermediate * hidden
    ffn_params_per_layer = gate_params + up_params + down_params

    # LayerNorm 파라미터 (무시해도 될 정도로 작음)
    norm_params_per_layer = hidden * 2  # input_layernorm + post_attention_layernorm

    total_per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer

    # 바이트 단위 계산 (BF16 = 2 bytes, INT8 = 1 byte, INT4 = 0.5 bytes)
    bytes_per_param = {"fp16": 2, "int8": 1, "int4": 0.5}

    layer_analysis = {
        "attention_params_per_layer": attn_params_per_layer,
        "ffn_params_per_layer": ffn_params_per_layer,
        "norm_params_per_layer": norm_params_per_layer,
        "total_params_per_layer": total_per_layer,
        "num_layers": n_layers,
    }

    # 하이브리드 양자화 시나리오들
    scenarios = {}

    # 시나리오 0: Full FP16 (baseline)
    full_fp16 = n_layers * total_per_layer * bytes_per_param["fp16"]
    scenarios["baseline_fp16"] = {
        "description": "전체 FP16 (baseline)",
        "attn_precision": "fp16",
        "ffn_precision": "fp16",
        "total_bytes": full_fp16,
        "total_gb": full_fp16 / 1024**3,
    }

    # 시나리오 1: 전체 INT4
    full_int4 = n_layers * total_per_layer * bytes_per_param["int4"]
    scenarios["full_int4"] = {
        "description": "전체 INT4",
        "attn_precision": "int4",
        "ffn_precision": "int4",
        "total_bytes": full_int4,
        "total_gb": full_int4 / 1024**3,
    }

    # 시나리오 2: Attention INT4 + FFN INT8
    hybrid_attn4_ffn8 = n_layers * (
        attn_params_per_layer * bytes_per_param["int4"] +
        ffn_params_per_layer * bytes_per_param["int8"] +
        norm_params_per_layer * bytes_per_param["fp16"]
    )
    scenarios["attn_int4_ffn_int8"] = {
        "description": "Attention INT4, FFN INT8",
        "attn_precision": "int4",
        "ffn_precision": "int8",
        "total_bytes": hybrid_attn4_ffn8,
        "total_gb": hybrid_attn4_ffn8 / 1024**3,
    }

    # 시나리오 3: Attention INT8 + FFN INT4
    hybrid_attn8_ffn4 = n_layers * (
        attn_params_per_layer * bytes_per_param["int8"] +
        ffn_params_per_layer * bytes_per_param["int4"] +
        norm_params_per_layer * bytes_per_param["fp16"]
    )
    scenarios["attn_int8_ffn_int4"] = {
        "description": "Attention INT8, FFN INT4",
        "attn_precision": "int8",
        "ffn_precision": "int4",
        "total_bytes": hybrid_attn8_ffn4,
        "total_gb": hybrid_attn8_ffn4 / 1024**3,
    }

    # 시나리오 4: 전체 INT8
    full_int8 = n_layers * total_per_layer * bytes_per_param["int8"]
    scenarios["full_int8"] = {
        "description": "전체 INT8",
        "attn_precision": "int8",
        "ffn_precision": "int8",
        "total_bytes": full_int8,
        "total_gb": full_int8 / 1024**3,
    }

    # 시나리오 5: 상위/하위 레이어 분리 (first N layers FP16, rest INT4)
    for split_at in [6, 12, 18, 24]:
        fp16_layers = n_layers - split_at  # 첫 N개는 INT4, 나머지는 FP16 (반대도 해볼 수 있음)
        int4_layers = split_at
        hybrid_split = (
            fp16_layers * total_per_layer * bytes_per_param["fp16"] +
            int4_layers * total_per_layer * bytes_per_param["int4"]
        )
        scenarios[f"first{split_at}_int4_rest_fp16"] = {
            "description": f"첫 {split_at}개 레이어 INT4, 나머지 FP16",
            "int4_layers": split_at,
            "fp16_layers": fp16_layers,
            "total_bytes": hybrid_split,
            "total_gb": hybrid_split / 1024**3,
        }

    results["layer_analysis"] = layer_analysis
    results["scenarios"] = scenarios

    return results


def quantization_simulation():
    """실제 텐서로 양자화 시뮬레이션 (정확도 영향 추정)."""

    results = {"quantization_error": {}}

    # 랜덤 텐서로 양자화 오차 시뮬레이션
    torch.manual_seed(42)

    for size_name, size in [("attention_weight", (4096, 4096)), ("ffn_weight", (4096, 14336))]:
        # FP16 원본 생성 (정규분포)
        original = torch.randn(size, dtype=torch.float16, device="cuda")

        # INT8 양자화 시뮬레이션 (per-channel absmax)
        scale_int8 = original.abs().max(dim=1, keepdim=True).values / 127.0
        quantized_int8 = torch.round(original / scale_int8).clamp(-128, 127)
        dequant_int8 = quantized_int8 * scale_int8
        error_int8 = (original - dequant_int8).abs().mean().item()
        relative_error_int8 = error_int8 / original.abs().mean().item()

        # INT4 양자화 시뮬레이션 (per-channel absmax, group_size=128)
        group_size = 128
        orig_reshaped = original.reshape(-1, group_size)
        scale_int4 = orig_reshaped.abs().max(dim=1, keepdim=True).values / 7.0
        quantized_int4 = torch.round(orig_reshaped / scale_int4).clamp(-8, 7)
        dequant_int4 = (quantized_int4 * scale_int4).reshape(size)
        error_int4 = (original - dequant_int4).abs().mean().item()
        relative_error_int4 = error_int4 / original.abs().mean().item()

        results["quantization_error"][size_name] = {
            "shape": list(size),
            "original_mean_abs": original.abs().mean().item(),
            "int8_abs_error": error_int8,
            "int8_relative_error": relative_error_int8,
            "int4_abs_error": error_int4,
            "int4_relative_error": relative_error_int4,
            "int4_vs_int8_error_ratio": relative_error_int4 / relative_error_int8 if relative_error_int8 > 0 else float("inf"),
        }

        del original, scale_int8, quantized_int8, dequant_int8
        del orig_reshaped, scale_int4, quantized_int4, dequant_int4

    torch.cuda.empty_cache()
    gc.collect()

    return results


def benchmark_quantized_matmul():
    """양자화된 행렬곱 속도 벤치마크 (FP16 vs simulated INT8 vs INT4)."""

    results = {"matmul_benchmark": {}}

    torch.manual_seed(42)
    torch.cuda.empty_cache()

    # 일반적인 Transformer 레이어 크기
    M = 256  # sequence length
    K = 4096  # hidden_size
    N = 4096  # output (attention)

    for name, (m, k, n) in [
        ("attention_proj", (M, K, K)),
        ("ffn_gate", (M, K, 14336)),
        ("ffn_down", (M, 14336, K)),
    ]:
        # FP16 baseline
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w_fp16 = torch.randn(n, k, dtype=torch.float16, device="cuda")

        # Warmup
        for _ in range(5):
            _ = x @ w_fp16.T
        torch.cuda.synchronize()

        # Benchmark FP16
        start = time.perf_counter()
        for _ in range(100):
            _ = x @ w_fp16.T
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / 100

        # INT8 시뮬레이션 (실제로는 FP16 연산, 메모리 절감 효과만 측정)
        scale = w_fp16.abs().max(dim=1, keepdim=True).values / 127.0
        w_int8 = torch.round(w_fp16 / scale).to(torch.int8)

        # Warmup
        for _ in range(5):
            w_dequant = w_int8.to(torch.float16) * scale
            _ = x @ w_dequant.T
        torch.cuda.synchronize()

        # Benchmark INT8 dequant + matmul
        start = time.perf_counter()
        for _ in range(100):
            w_dequant = w_int8.to(torch.float16) * scale
            _ = x @ w_dequant.T
        torch.cuda.synchronize()
        int8_time = (time.perf_counter() - start) / 100

        mem_fp16 = w_fp16.nelement() * 2 / 1024**2
        mem_int8 = (w_int8.nelement() * 1 + scale.nelement() * 2) / 1024**2

        results["matmul_benchmark"][name] = {
            "shape": f"{m}x{k} @ {k}x{n}",
            "fp16_time_ms": fp16_time * 1000,
            "int8_dequant_time_ms": int8_time * 1000,
            "speedup_or_slowdown": fp16_time / int8_time if int8_time > 0 else 0,
            "weight_mem_fp16_mb": mem_fp16,
            "weight_mem_int8_mb": mem_int8,
            "memory_saving_ratio": mem_int8 / mem_fp16,
        }

        del x, w_fp16, w_int8, scale, w_dequant
        torch.cuda.empty_cache()

    return results


def module_level_offloading_analysis():
    """모듈별 오프로딩 + 양자화 조합 분석."""

    # Alpamayo 모듈별 크기 (기존 연구 결과 기반)
    modules = {
        "vision_encoder": {"fp16_gb": 1.15, "description": "SigLIP Vision Encoder"},
        "vlm_8b": {"fp16_gb": 15.17, "description": "Qwen3-VL 8B (36 layers)"},
        "expert_diffusion": {"fp16_gb": 4.56, "description": "Expert + Action + Diffusion"},
    }

    # 각 모듈에 대한 양자화 전략
    strategies = []

    # 전략 1: Vision FP16(CPU→GPU), VLM INT4(GPU), Expert FP16(CPU→GPU)
    strategies.append({
        "name": "VLM-INT4 + Others-FP16-Offload",
        "vision": {"precision": "fp16", "location": "cpu_to_gpu", "gb": 1.15},
        "vlm": {"precision": "int4", "location": "gpu", "gb": 15.17 / 4},
        "expert": {"precision": "fp16", "location": "cpu_to_gpu", "gb": 4.56},
        "peak_vram_gb": max(1.15, 15.17/4, 4.56) + 1.0,  # +1 for KV cache/activations
        "description": "VLM만 INT4, 나머지는 FP16으로 필요 시 GPU 전송"
    })

    # 전략 2: Vision FP16(CPU→GPU), VLM INT4(GPU), Expert INT8(GPU)
    strategies.append({
        "name": "VLM-INT4 + Expert-INT8",
        "vision": {"precision": "fp16", "location": "cpu_to_gpu", "gb": 1.15},
        "vlm": {"precision": "int4", "location": "gpu", "gb": 15.17 / 4},
        "expert": {"precision": "int8", "location": "gpu", "gb": 4.56 / 2},
        "peak_vram_gb": 15.17/4 + 4.56/2 + 1.0,
        "description": "VLM INT4 + Expert INT8, Vision 온디맨드"
    })

    # 전략 3: 모든 것 INT4
    strategies.append({
        "name": "All-INT4",
        "vision": {"precision": "int4", "location": "gpu", "gb": 1.15 / 4},
        "vlm": {"precision": "int4", "location": "gpu", "gb": 15.17 / 4},
        "expert": {"precision": "int4", "location": "gpu", "gb": 4.56 / 4},
        "peak_vram_gb": (1.15 + 15.17 + 4.56) / 4 + 1.0,
        "description": "전체 INT4 양자화"
    })

    # 전략 4: VLM Attention INT4 + FFN INT8 + Expert FP16(offload)
    # VLM에서 Attention 비율 약 30%, FFN 비율 약 67% (나머지 norm 등)
    attn_ratio = 0.30
    ffn_ratio = 0.67
    vlm_hybrid = 15.17 * (attn_ratio / 4 + ffn_ratio / 2 + (1 - attn_ratio - ffn_ratio))
    strategies.append({
        "name": "VLM-Hybrid(Attn-INT4+FFN-INT8)",
        "vision": {"precision": "fp16", "location": "cpu_to_gpu", "gb": 1.15},
        "vlm": {"precision": "hybrid", "location": "gpu", "gb": vlm_hybrid},
        "expert": {"precision": "fp16", "location": "cpu_to_gpu", "gb": 4.56},
        "peak_vram_gb": vlm_hybrid + 1.5,
        "description": "VLM Attention INT4 + FFN INT8, 나머지 FP16 오프로드"
    })

    # 전략 5: VLM INT4 + Expert FP16 + Vision FP16 모두 동시 GPU
    strategies.append({
        "name": "All-on-GPU(VLM-INT4)",
        "vision": {"precision": "fp16", "location": "gpu", "gb": 1.15},
        "vlm": {"precision": "int4", "location": "gpu", "gb": 15.17 / 4},
        "expert": {"precision": "fp16", "location": "gpu", "gb": 4.56},
        "peak_vram_gb": 1.15 + 15.17/4 + 4.56 + 1.5,
        "description": "VLM INT4로 전체 GPU 탑재 시도"
    })

    return {"offloading_strategies": strategies}


def main():
    print("=" * 70)
    print("실험 1: 하이브리드 양자화 (Mixed Precision Offloading) 분석")
    print("=" * 70)

    all_results = {}

    # 1. 모델 구조 분석
    print("\n[1/4] 모델 구조 및 양자화 시나리오 분석...")
    structure_results = analyze_model_structure()
    all_results.update(structure_results)

    print(f"  VLM Hidden Size: {structure_results['model_info']['hidden_size']}")
    print(f"  VLM Intermediate Size: {structure_results['model_info']['intermediate_size']}")
    print(f"  VLM Layers: {structure_results['model_info']['num_hidden_layers']}")
    print(f"  Attention params/layer: {structure_results['layer_analysis']['attention_params_per_layer']:,}")
    print(f"  FFN params/layer: {structure_results['layer_analysis']['ffn_params_per_layer']:,}")
    print()

    for name, scenario in structure_results["scenarios"].items():
        print(f"  {scenario['description']}: {scenario['total_gb']:.2f} GB")

    # 2. 양자화 오차 시뮬레이션
    print("\n[2/4] 양자화 오차 시뮬레이션...")
    quant_results = quantization_simulation()
    all_results.update(quant_results)

    for name, data in quant_results["quantization_error"].items():
        print(f"  {name}:")
        print(f"    INT8 relative error: {data['int8_relative_error']:.4f}")
        print(f"    INT4 relative error: {data['int4_relative_error']:.4f}")
        print(f"    INT4/INT8 ratio: {data['int4_vs_int8_error_ratio']:.2f}x")

    # 3. 행렬곱 벤치마크
    print("\n[3/4] 양자화 행렬곱 속도 벤치마크...")
    matmul_results = benchmark_quantized_matmul()
    all_results.update(matmul_results)

    for name, data in matmul_results["matmul_benchmark"].items():
        print(f"  {name} ({data['shape']}):")
        print(f"    FP16: {data['fp16_time_ms']:.3f} ms")
        print(f"    INT8 dequant: {data['int8_dequant_time_ms']:.3f} ms")
        print(f"    Memory saving: {data['memory_saving_ratio']:.2%}")

    # 4. 모듈별 오프로딩 + 양자화 조합
    print("\n[4/4] 모듈별 오프로딩 + 양자화 조합 분석...")
    offload_results = module_level_offloading_analysis()
    all_results.update(offload_results)

    print(f"\n  {'전략':<45} {'Peak VRAM (GB)':>15}  {'12GB 이내':>10}")
    print("  " + "-" * 75)
    for strategy in offload_results["offloading_strategies"]:
        fits = "OK" if strategy["peak_vram_gb"] <= 12.0 else "OVER"
        print(f"  {strategy['name']:<45} {strategy['peak_vram_gb']:>12.2f} GB  {fits:>10}")

    # 결과 저장
    output_path = "/home/seungwoo/workspace/research/04-creative-approaches/exp01_hybrid_quant_results.json"

    # JSON 직렬화 호환을 위한 변환
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    print(f"\n결과 저장: {output_path}")
    print("실험 1 완료!")


if __name__ == "__main__":
    main()
