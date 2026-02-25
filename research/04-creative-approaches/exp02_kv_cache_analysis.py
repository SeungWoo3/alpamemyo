#!/usr/bin/env python3
"""실험 2: KV Cache 압축/오프로딩 분석

VLM의 autoregressive 생성에서 KV cache 메모리 증가를 실측하고,
압축/오프로딩 가능성을 분석한다.
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

def analyze_kv_cache_theory():
    """KV cache 메모리 이론적 분석."""
    from transformers import Qwen3VLConfig

    config = Qwen3VLConfig.from_pretrained("nvidia/Alpamayo-R1-10B")
    tc = config.text_config

    hidden_size = tc.hidden_size
    num_layers = tc.num_hidden_layers
    num_kv_heads = tc.num_key_value_heads
    head_dim = getattr(tc, "head_dim", hidden_size // tc.num_attention_heads)

    # KV cache per token (2 bytes per param for BF16)
    # Each layer: K + V = 2 * num_kv_heads * head_dim * 2 (bytes, BF16)
    kv_per_token_per_layer = 2 * num_kv_heads * head_dim * 2  # bytes
    kv_per_token_total = kv_per_token_per_layer * num_layers  # bytes

    results = {
        "kv_cache_config": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "kv_per_token_per_layer_bytes": kv_per_token_per_layer,
        "kv_per_token_total_bytes": kv_per_token_total,
        "kv_per_token_total_kb": kv_per_token_total / 1024,
        "kv_per_token_total_mb": kv_per_token_total / 1024**2,
    }

    # 토큰 수에 따른 KV cache 크기 예측
    token_counts = [64, 128, 256, 512, 1024, 2048, 4096]
    kv_cache_sizes = {}
    for n_tokens in token_counts:
        size_mb = n_tokens * kv_per_token_total / 1024**2
        kv_cache_sizes[str(n_tokens)] = {
            "tokens": n_tokens,
            "kv_cache_mb": round(size_mb, 2),
            "kv_cache_gb": round(size_mb / 1024, 3),
        }

    results["kv_cache_by_token_count"] = kv_cache_sizes

    # Alpamayo 입력 토큰 수 추정
    # Vision: 7 프레임 * 각 ~588 토큰 (320x576 해상도) ≈ 4116 vision 토큰
    # + trajectory 토큰 (48) + text 토큰 (~50) + generated CoT (~200)
    alpamayo_input_tokens = 4116 + 48 + 50
    alpamayo_max_tokens = alpamayo_input_tokens + 256  # max_generation_length=256

    results["alpamayo_token_estimate"] = {
        "vision_tokens": 4116,
        "traj_tokens": 48,
        "text_tokens": 50,
        "total_input_tokens": alpamayo_input_tokens,
        "max_total_tokens": alpamayo_max_tokens,
        "estimated_kv_cache_mb": round(alpamayo_max_tokens * kv_per_token_total / 1024**2, 2),
        "estimated_kv_cache_gb": round(alpamayo_max_tokens * kv_per_token_total / 1024**3, 3),
    }

    return results


def simulate_kv_cache_memory():
    """실제 텐서로 KV cache 메모리 증가를 실측."""

    results = {"kv_cache_simulation": {}}

    torch.cuda.empty_cache()
    gc.collect()

    # Qwen3-VL 8B 파라미터 사용
    num_layers = 36
    num_kv_heads = 4  # GQA: 4 KV heads
    head_dim = 128
    batch_size = 1
    dtype = torch.bfloat16

    measurements = []
    token_counts = [64, 128, 256, 512, 1024, 2048, 3072, 4096]

    for n_tokens in token_counts:
        torch.cuda.empty_cache()
        gc.collect()

        base_mem = get_gpu_memory_mb()

        # KV cache 할당 시뮬레이션
        kv_caches = []
        for layer_idx in range(num_layers):
            k_cache = torch.randn(batch_size, num_kv_heads, n_tokens, head_dim,
                                  dtype=dtype, device="cuda")
            v_cache = torch.randn(batch_size, num_kv_heads, n_tokens, head_dim,
                                  dtype=dtype, device="cuda")
            kv_caches.append((k_cache, v_cache))

        after_mem = get_gpu_memory_mb()
        actual_kv_mb = after_mem - base_mem

        # 이론값
        theory_kv_mb = (num_layers * 2 * batch_size * num_kv_heads * n_tokens * head_dim * 2) / 1024**2

        measurements.append({
            "tokens": n_tokens,
            "actual_kv_mb": round(actual_kv_mb, 2),
            "theory_kv_mb": round(theory_kv_mb, 2),
            "match_ratio": round(actual_kv_mb / theory_kv_mb, 3) if theory_kv_mb > 0 else 0,
        })

        print(f"  {n_tokens:5d} tokens: actual={actual_kv_mb:8.2f} MB, theory={theory_kv_mb:8.2f} MB")

        del kv_caches
        torch.cuda.empty_cache()

    results["kv_cache_simulation"] = measurements
    return results


def kv_cache_compression_analysis():
    """KV cache 압축 기법 분석 및 시뮬레이션."""

    results = {"compression_methods": {}}

    torch.cuda.empty_cache()
    gc.collect()

    # 대표적인 KV cache 텐서 (1 layer, 1024 tokens)
    num_kv_heads = 4
    seq_len = 1024
    head_dim = 128
    dtype = torch.bfloat16

    k_cache = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v_cache = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device="cuda")

    original_size_bytes = (k_cache.nelement() + v_cache.nelement()) * 2  # BF16 = 2 bytes

    # 방법 1: INT8 양자화
    k_fp32 = k_cache.float()
    v_fp32 = v_cache.float()

    k_scale = k_fp32.abs().amax(dim=-1, keepdim=True) / 127
    v_scale = v_fp32.abs().amax(dim=-1, keepdim=True) / 127

    k_int8 = (k_fp32 / k_scale).round().clamp(-128, 127).to(torch.int8)
    v_int8 = (v_fp32 / v_scale).round().clamp(-128, 127).to(torch.int8)

    int8_size = (k_int8.nelement() + v_int8.nelement()) * 1 + (k_scale.nelement() + v_scale.nelement()) * 4
    int8_error_k = ((k_int8.float() * k_scale - k_fp32).abs() / (k_fp32.abs() + 1e-8)).mean().item()
    int8_error_v = ((v_int8.float() * v_scale - v_fp32).abs() / (v_fp32.abs() + 1e-8)).mean().item()

    results["compression_methods"]["int8_quantization"] = {
        "description": "KV cache INT8 per-head-dim 양자화",
        "original_size_bytes": original_size_bytes,
        "compressed_size_bytes": int8_size,
        "compression_ratio": original_size_bytes / int8_size,
        "relative_error_k": round(int8_error_k, 6),
        "relative_error_v": round(int8_error_v, 6),
        "savings_percent": round((1 - int8_size / original_size_bytes) * 100, 1),
    }

    del k_int8, v_int8, k_scale, v_scale

    # 방법 2: INT4 양자화 (group quantization)
    group_size = 32
    k_reshaped = k_fp32.reshape(-1, group_size)
    v_reshaped = v_fp32.reshape(-1, group_size)

    k_scale4 = k_reshaped.abs().amax(dim=-1, keepdim=True) / 7
    v_scale4 = v_reshaped.abs().amax(dim=-1, keepdim=True) / 7

    k_int4 = (k_reshaped / k_scale4).round().clamp(-8, 7)
    v_int4 = (v_reshaped / v_scale4).round().clamp(-8, 7)

    int4_data_bytes = (k_int4.nelement() + v_int4.nelement()) // 2  # 4 bits
    int4_scale_bytes = (k_scale4.nelement() + v_scale4.nelement()) * 4
    int4_size = int4_data_bytes + int4_scale_bytes

    int4_error_k = ((k_int4 * k_scale4 - k_reshaped).abs() / (k_reshaped.abs() + 1e-8)).mean().item()
    int4_error_v = ((v_int4 * v_scale4 - v_reshaped).abs() / (v_reshaped.abs() + 1e-8)).mean().item()

    results["compression_methods"]["int4_quantization"] = {
        "description": f"KV cache INT4 group 양자화 (group_size={group_size})",
        "original_size_bytes": original_size_bytes,
        "compressed_size_bytes": int4_size,
        "compression_ratio": original_size_bytes / int4_size,
        "relative_error_k": round(int4_error_k, 6),
        "relative_error_v": round(int4_error_v, 6),
        "savings_percent": round((1 - int4_size / original_size_bytes) * 100, 1),
    }

    del k_int4, v_int4, k_scale4, v_scale4, k_reshaped, v_reshaped

    # 방법 3: Token Eviction (중요하지 않은 토큰의 KV 제거)
    # Attention score 기반 시뮬레이션
    # 실제로는 attention score를 기반으로 가장 적게 참조되는 토큰을 제거
    for keep_ratio in [0.5, 0.25]:
        keep_tokens = int(seq_len * keep_ratio)
        evicted_size = original_size_bytes * keep_ratio

        # 시뮬레이션: 랜덤하게 토큰을 선택 (실제로는 attention score 기반)
        indices = torch.randperm(seq_len, device="cuda")[:keep_tokens].sort().values
        k_evicted = k_cache[:, :, indices, :]
        v_evicted = v_cache[:, :, indices, :]

        results["compression_methods"][f"token_eviction_{int(keep_ratio*100)}pct"] = {
            "description": f"Token Eviction (상위 {int(keep_ratio*100)}% 토큰만 유지)",
            "original_tokens": seq_len,
            "kept_tokens": keep_tokens,
            "original_size_bytes": original_size_bytes,
            "compressed_size_bytes": int(evicted_size),
            "compression_ratio": 1.0 / keep_ratio,
            "savings_percent": round((1 - keep_ratio) * 100, 1),
            "note": "품질 영향은 attention score 분포에 따라 다름",
        }

    # 방법 4: CPU 오프로딩 + Async prefetch
    # KV cache를 CPU로 이동하는 시간 측정
    torch.cuda.synchronize()

    # Pinned memory 사용
    k_cpu_pinned = torch.empty_like(k_cache, device="cpu").pin_memory()
    v_cpu_pinned = torch.empty_like(v_cache, device="cpu").pin_memory()

    # GPU -> CPU (D2H)
    torch.cuda.synchronize()
    start = time.perf_counter()
    k_cpu_pinned.copy_(k_cache, non_blocking=False)
    v_cpu_pinned.copy_(v_cache, non_blocking=False)
    torch.cuda.synchronize()
    d2h_time = time.perf_counter() - start

    # CPU -> GPU (H2D)
    torch.cuda.synchronize()
    start = time.perf_counter()
    k_cache.copy_(k_cpu_pinned, non_blocking=False)
    v_cache.copy_(v_cpu_pinned, non_blocking=False)
    torch.cuda.synchronize()
    h2d_time = time.perf_counter() - start

    transfer_size_mb = original_size_bytes / 1024**2

    results["compression_methods"]["cpu_offloading"] = {
        "description": "KV cache CPU 오프로딩 (pinned memory)",
        "transfer_size_mb": round(transfer_size_mb, 2),
        "d2h_time_ms": round(d2h_time * 1000, 2),
        "h2d_time_ms": round(h2d_time * 1000, 2),
        "d2h_bandwidth_gbps": round(transfer_size_mb / 1024 / d2h_time, 2),
        "h2d_bandwidth_gbps": round(transfer_size_mb / 1024 / h2d_time, 2),
        "note": "1 layer 기준. 36 layers = 약 36x 시간",
    }

    del k_cache, v_cache, k_cpu_pinned, v_cpu_pinned, k_fp32, v_fp32
    torch.cuda.empty_cache()

    return results


def kv_cache_offload_pipeline_simulation():
    """KV cache 오프로딩 파이프라인 시뮬레이션: 레이어별 KV 이동 시간 측정."""

    results = {"pipeline_simulation": {}}

    torch.cuda.empty_cache()
    gc.collect()

    num_layers = 36
    num_kv_heads = 4
    head_dim = 128
    seq_len = 1024  # Alpamayo는 약 4000+ 토큰 사용
    dtype = torch.bfloat16

    # 전체 KV cache를 CPU에 둔 상태에서 레이어별로 GPU로 전송
    kv_cpu = []
    for _ in range(num_layers):
        k = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype).pin_memory()
        v = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=dtype).pin_memory()
        kv_cpu.append((k, v))

    # GPU 버퍼
    k_gpu = torch.empty(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device="cuda")
    v_gpu = torch.empty(1, num_kv_heads, seq_len, head_dim, dtype=dtype, device="cuda")

    # 순차 전송 시간 측정
    torch.cuda.synchronize()
    start = time.perf_counter()
    for layer_idx in range(num_layers):
        k_gpu.copy_(kv_cpu[layer_idx][0], non_blocking=False)
        v_gpu.copy_(kv_cpu[layer_idx][1], non_blocking=False)
    torch.cuda.synchronize()
    sequential_time = time.perf_counter() - start

    # 비동기 전송 (다음 레이어 프리페치) 시간 측정
    stream = torch.cuda.Stream()
    k_gpu2 = torch.empty_like(k_gpu)
    v_gpu2 = torch.empty_like(v_gpu)

    torch.cuda.synchronize()
    start = time.perf_counter()
    # 첫 레이어 로드
    k_gpu.copy_(kv_cpu[0][0], non_blocking=False)
    v_gpu.copy_(kv_cpu[0][1], non_blocking=False)

    for layer_idx in range(1, num_layers):
        # 비동기로 다음 레이어 프리페치
        with torch.cuda.stream(stream):
            k_gpu2.copy_(kv_cpu[layer_idx][0], non_blocking=True)
            v_gpu2.copy_(kv_cpu[layer_idx][1], non_blocking=True)

        # 현재 레이어 "처리" 시뮬레이션 (간단한 연산)
        _ = k_gpu.sum() + v_gpu.sum()

        # 다음 레이어 전송 완료 대기
        stream.synchronize()

        # 버퍼 스왑
        k_gpu, k_gpu2 = k_gpu2, k_gpu
        v_gpu, v_gpu2 = v_gpu2, v_gpu

    torch.cuda.synchronize()
    async_time = time.perf_counter() - start

    per_layer_bytes = 2 * num_kv_heads * seq_len * head_dim * 2
    total_bytes = per_layer_bytes * num_layers

    results["pipeline_simulation"] = {
        "config": {
            "num_layers": num_layers,
            "seq_len": seq_len,
            "per_layer_kv_mb": round(per_layer_bytes / 1024**2, 2),
            "total_kv_mb": round(total_bytes / 1024**2, 2),
        },
        "sequential_transfer_ms": round(sequential_time * 1000, 2),
        "async_pipeline_ms": round(async_time * 1000, 2),
        "speedup": round(sequential_time / async_time, 2) if async_time > 0 else 0,
        "effective_bandwidth_sequential_gbps": round(total_bytes / 1024**3 / sequential_time, 2),
        "effective_bandwidth_async_gbps": round(total_bytes / 1024**3 / async_time, 2),
    }

    del kv_cpu, k_gpu, v_gpu, k_gpu2, v_gpu2
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    print("=" * 70)
    print("실험 2: KV Cache 압축/오프로딩 분석")
    print("=" * 70)

    all_results = {}

    # 1. 이론적 분석
    print("\n[1/4] KV Cache 이론적 분석...")
    theory_results = analyze_kv_cache_theory()
    all_results.update(theory_results)

    kv_cfg = theory_results["kv_cache_config"]
    print(f"  KV heads: {kv_cfg['num_kv_heads']}, Head dim: {kv_cfg['head_dim']}")
    print(f"  KV cache per token (all layers): {theory_results['kv_per_token_total_kb']:.2f} KB")

    for n, data in theory_results["kv_cache_by_token_count"].items():
        print(f"  {data['tokens']:5d} tokens -> {data['kv_cache_mb']:8.2f} MB ({data['kv_cache_gb']:.3f} GB)")

    est = theory_results["alpamayo_token_estimate"]
    print(f"\n  Alpamayo 추정 총 토큰: {est['max_total_tokens']}")
    print(f"  Alpamayo 추정 KV cache: {est['estimated_kv_cache_mb']:.2f} MB ({est['estimated_kv_cache_gb']:.3f} GB)")

    # 2. 실제 KV cache 메모리 측정
    print("\n[2/4] KV Cache 메모리 실측...")
    sim_results = simulate_kv_cache_memory()
    all_results.update(sim_results)

    # 3. KV cache 압축 기법 분석
    print("\n[3/4] KV Cache 압축 기법 분석...")
    compression_results = kv_cache_compression_analysis()
    all_results.update(compression_results)

    print(f"\n  {'방법':<40} {'압축률':>8} {'오차(K)':>10} {'절감':>8}")
    print("  " + "-" * 70)
    for name, data in compression_results["compression_methods"].items():
        error_str = f"{data.get('relative_error_k', 'N/A')}" if 'relative_error_k' in data else "N/A"
        cr = data.get('compression_ratio', 0)
        sp = data.get('savings_percent', 0)
        print(f"  {data['description']:<40} {cr:>6.2f}x {error_str:>10} {sp:>6.1f}%")

    # 4. 오프로딩 파이프라인 시뮬레이션
    print("\n[4/4] KV Cache 오프로딩 파이프라인 시뮬레이션...")
    pipeline_results = kv_cache_offload_pipeline_simulation()
    all_results.update(pipeline_results)

    ps = pipeline_results["pipeline_simulation"]
    print(f"  Total KV cache: {ps['config']['total_kv_mb']:.2f} MB")
    print(f"  Sequential transfer: {ps['sequential_transfer_ms']:.2f} ms")
    print(f"  Async pipeline: {ps['async_pipeline_ms']:.2f} ms (speedup: {ps['speedup']:.2f}x)")

    # 결과 저장
    output_path = "/home/seungwoo/workspace/research/04-creative-approaches/exp02_kv_cache_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else int(x) if isinstance(x, (np.integer,)) else x)

    print(f"\n결과 저장: {output_path}")
    print("실험 2 완료!")


if __name__ == "__main__":
    main()
