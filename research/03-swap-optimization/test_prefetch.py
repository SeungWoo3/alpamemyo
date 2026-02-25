#!/usr/bin/env python3
"""Part 2: cudaMemPrefetchAsync 프리페칭 실험

CUDA Unified Memory의 cudaMemPrefetchAsync를 사용하여:
1. Unified Memory 할당 후 사전 로딩 시도
2. 프리페칭 vs 미프리페칭 접근 시간 비교
3. 프리페치 크기/패턴별 효과 분석

NOTE: WSL2에서는 cudaMemPrefetchAsync(device=-1, CPU)가
"invalid device ordinal" 에러를 발생시킬 수 있다.
이 경우 cudaMemcpyAsync 기반 대안 실험으로 전환한다.
"""

import os
import sys
import json
import time
import gc
import ctypes
from datetime import datetime

import torch
import numpy as np

OUTPUT_DIR = "/home/seungwoo/workspace/research/03-swap-optimization"
PYTHON_ENV = "/home/seungwoo/workspace/alpamayo/ar1_venv"
CUDA_RT_PATH = f"{PYTHON_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"

cuda_rt = ctypes.cdll.LoadLibrary(CUDA_RT_PATH)

# CUDA constants
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# ============================================================
# Helper: Check if prefetch to CPU works on this platform
# ============================================================

def check_prefetch_cpu_support():
    """WSL2에서 cudaMemPrefetchAsync(cpu) 지원 여부를 확인한다."""
    ptr = ctypes.c_void_p()
    ret = cuda_rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(4096), ctypes.c_uint(1))
    if ret != 0:
        return False

    # Try prefetch to CPU (device=-1)
    ret = cuda_rt.cudaMemPrefetchAsync(ptr, ctypes.c_size_t(4096), ctypes.c_int(-1), ctypes.c_void_p(0))
    cuda_rt.cudaFree(ptr)

    if ret != 0:
        # Clear the error
        cuda_rt.cudaGetLastError()
        return False
    return True


# ============================================================
# Experiment 1: Unified Memory Prefetch to GPU Only
# ============================================================

def experiment_prefetch_to_gpu():
    """cudaMemPrefetchAsync로 GPU 프리페치 효과를 측정한다.
    WSL2에서는 CPU 프리페치가 불가능하므로, GPU 프리페치만 테스트한다.
    대신 cudaMemset으로 초기 GPU 접근을 트리거하고, 재접근 시간을 비교한다.
    """
    print("=" * 70)
    print("Experiment 1: Unified Memory Prefetch Analysis (GPU-only)")
    print("=" * 70)

    prefetch_cpu_ok = check_prefetch_cpu_support()
    print(f"\n  cudaMemPrefetchAsync to CPU supported: {prefetch_cpu_ok}")

    results = []

    for size_mb in [64, 128, 256, 512, 1024, 2048]:
        size_bytes = size_mb * 1024 * 1024

        print(f"\n--- Test size: {size_mb} MB ---")

        # Allocate managed memory
        ptr = ctypes.c_void_p()
        ret = cuda_rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size_bytes), ctypes.c_uint(1))
        if ret != 0:
            print(f"  cudaMallocManaged FAILED (code={ret})")
            results.append({"size_mb": size_mb, "error": f"alloc failed code={ret}"})
            continue

        # Test 1: First GPU access (cold - data on CPU initially)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        cuda_rt.cudaMemset(ptr, ctypes.c_int(0), ctypes.c_size_t(size_bytes))
        end.record()
        torch.cuda.synchronize()
        first_access_ms = start.elapsed_time(end)

        # Test 2: Second GPU access (warm - data already on GPU)
        start.record()
        cuda_rt.cudaMemset(ptr, ctypes.c_int(1), ctypes.c_size_t(size_bytes))
        end.record()
        torch.cuda.synchronize()
        second_access_ms = start.elapsed_time(end)

        # Test 3: Prefetch to GPU after fresh allocation
        cuda_rt.cudaFree(ptr)
        ptr = ctypes.c_void_p()
        ret = cuda_rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size_bytes), ctypes.c_uint(1))
        if ret != 0:
            continue

        # Prefetch to GPU first
        start.record()
        cuda_rt.cudaMemPrefetchAsync(ptr, ctypes.c_size_t(size_bytes), ctypes.c_int(0), ctypes.c_void_p(0))
        end.record()
        torch.cuda.synchronize()
        prefetch_ms = start.elapsed_time(end)

        # Then access (should be fast)
        start.record()
        cuda_rt.cudaMemset(ptr, ctypes.c_int(2), ctypes.c_size_t(size_bytes))
        end.record()
        torch.cuda.synchronize()
        after_prefetch_ms = start.elapsed_time(end)

        cold_bw = size_bytes / 1e9 / max(first_access_ms / 1000, 1e-9)
        warm_bw = size_bytes / 1e9 / max(second_access_ms / 1000, 1e-9)
        prefetch_bw = size_bytes / 1e9 / max(prefetch_ms / 1000, 1e-9)

        print(f"  Cold access (first): {first_access_ms:.3f} ms ({cold_bw:.2f} GB/s)")
        print(f"  Warm access (second): {second_access_ms:.3f} ms ({warm_bw:.2f} GB/s)")
        print(f"  Prefetch to GPU: {prefetch_ms:.3f} ms ({prefetch_bw:.2f} GB/s)")
        print(f"  Post-prefetch access: {after_prefetch_ms:.3f} ms")
        print(f"  Cold/Warm ratio: {first_access_ms / max(second_access_ms, 0.001):.2f}x")

        results.append({
            "size_mb": size_mb,
            "cold_access_ms": round(first_access_ms, 3),
            "cold_bw_gbs": round(cold_bw, 3),
            "warm_access_ms": round(second_access_ms, 3),
            "warm_bw_gbs": round(warm_bw, 3),
            "prefetch_ms": round(prefetch_ms, 3),
            "prefetch_bw_gbs": round(prefetch_bw, 3),
            "post_prefetch_ms": round(after_prefetch_ms, 3),
            "cold_warm_ratio": round(first_access_ms / max(second_access_ms, 0.001), 2),
        })

        cuda_rt.cudaFree(ptr)

    return {"prefetch_cpu_supported": prefetch_cpu_ok, "data": results}


# ============================================================
# Experiment 2: Prefetch Overlap with Computation
# ============================================================

def experiment_prefetch_overlap():
    """프리페치와 연산의 중첩 가능성을 테스트한다.
    PyTorch 텐서를 사용하여 GPU 프리페치와 연산 중첩을 측정한다."""
    print("\n" + "=" * 70)
    print("Experiment 2: Prefetch Overlap with Computation")
    print("=" * 70)

    results = []

    # Reset GPU state
    torch.cuda.empty_cache()
    gc.collect()

    # Create a compute-heavy task on GPU
    compute_size = 2048

    try:
        A = torch.randn(compute_size, compute_size, device="cuda", dtype=torch.float32)
        B = torch.randn(compute_size, compute_size, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  Failed to create matrices: {e}")
        # Reset CUDA error state
        torch.cuda.empty_cache()
        gc.collect()
        A = torch.randn(compute_size, compute_size, device="cuda", dtype=torch.float32)
        B = torch.randn(compute_size, compute_size, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()

    # Measure compute time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(3):
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        del C

    start.record()
    C = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    compute_ms = start.elapsed_time(end)
    print(f"\nBaseline matmul ({compute_size}x{compute_size}): {compute_ms:.3f} ms")
    del C

    # Test prefetch overlap using PyTorch tensors (pinned memory transfer)
    for transfer_mb in [128, 256, 512, 1024]:
        n_elements = transfer_mb * 1024 * 1024 // 2  # FP16
        transfer_bytes = transfer_mb * 1024 * 1024

        print(f"\n--- Transfer: {transfer_mb} MB + Matmul ---")

        try:
            cpu_tensor = torch.randn(n_elements, dtype=torch.float16).pin_memory()

            # Test 1: Sequential (transfer then compute)
            torch.cuda.empty_cache()
            start.record()
            gpu_t = cpu_tensor.to("cuda", non_blocking=True)
            torch.cuda.synchronize()  # Wait for transfer
            C = torch.matmul(A, B)
            end.record()
            torch.cuda.synchronize()
            sequential_ms = start.elapsed_time(end)
            del C, gpu_t

            # Test 2: Overlapped using separate streams
            torch.cuda.empty_cache()
            transfer_stream = torch.cuda.Stream()

            start.record()
            # Start transfer on separate stream
            with torch.cuda.stream(transfer_stream):
                gpu_t = cpu_tensor.to("cuda", non_blocking=True)

            # Compute on default stream simultaneously
            C = torch.matmul(A, B)

            # Wait for both
            torch.cuda.current_stream().wait_stream(transfer_stream)
            end.record()
            torch.cuda.synchronize()
            overlap_ms = start.elapsed_time(end)
            del C, gpu_t

            # Test 3: Transfer only
            torch.cuda.empty_cache()
            start.record()
            gpu_t = cpu_tensor.to("cuda", non_blocking=True)
            end.record()
            torch.cuda.synchronize()
            transfer_only_ms = start.elapsed_time(end)
            del gpu_t

            overlap_ratio = sequential_ms / max(overlap_ms, 0.001)
            savings_pct = (sequential_ms - overlap_ms) / sequential_ms * 100

            print(f"  Transfer only: {transfer_only_ms:.3f} ms")
            print(f"  Sequential (transfer + compute): {sequential_ms:.3f} ms")
            print(f"  Overlapped (stream-based): {overlap_ms:.3f} ms")
            print(f"  Speedup: {overlap_ratio:.2f}x")
            print(f"  Time savings: {savings_pct:.1f}%")

            results.append({
                "transfer_mb": transfer_mb,
                "compute_only_ms": round(compute_ms, 3),
                "transfer_only_ms": round(transfer_only_ms, 3),
                "sequential_ms": round(sequential_ms, 3),
                "overlapped_ms": round(overlap_ms, 3),
                "speedup": round(overlap_ratio, 3),
                "savings_pct": round(savings_pct, 1),
            })

            del cpu_tensor
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"transfer_mb": transfer_mb, "error": str(e)[:200]})

        torch.cuda.empty_cache()
        gc.collect()

    del A, B
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Experiment 3: Prefetch Granularity Analysis
# ============================================================

def experiment_prefetch_granularity():
    """프리페치 단위 크기에 따른 효율성을 분석한다.
    PyTorch pinned tensor를 사용하여 전송 단위별 효과를 측정한다."""
    print("\n" + "=" * 70)
    print("Experiment 3: Transfer Granularity Analysis")
    print("=" * 70)

    results = []
    total_size_mb = 512  # Total 512MB to transfer
    total_bytes = total_size_mb * 1024 * 1024
    n_trials = 3

    for chunk_mb in [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        chunk_bytes = int(chunk_mb * 1024 * 1024)
        chunk_elements = chunk_bytes // 2  # FP16
        n_chunks = total_bytes // chunk_bytes

        print(f"\n--- {total_size_mb}MB as {n_chunks} x {chunk_mb}MB chunks ---")

        try:
            # Create pinned CPU chunks
            chunks = [torch.randn(chunk_elements, dtype=torch.float16).pin_memory() for _ in range(n_chunks)]

            h2d_times = []
            for trial in range(n_trials):
                torch.cuda.empty_cache()
                gc.collect()

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                gpu_chunks = []
                start.record()
                for chunk in chunks:
                    gpu_chunk = chunk.to("cuda", non_blocking=True)
                    gpu_chunks.append(gpu_chunk)
                end.record()
                torch.cuda.synchronize()
                h2d_times.append(start.elapsed_time(end))

                for g in gpu_chunks:
                    del g
                del gpu_chunks
                torch.cuda.empty_cache()

            avg_h2d = np.mean(h2d_times)
            h2d_bw = total_bytes / 1e9 / (avg_h2d / 1000) if avg_h2d > 0 else 0

            print(f"  H2D: {avg_h2d:.3f} ms ({h2d_bw:.2f} GB/s)")

            results.append({
                "chunk_mb": chunk_mb,
                "n_chunks": n_chunks,
                "h2d_ms": round(avg_h2d, 3),
                "h2d_bw_gbs": round(h2d_bw, 3),
            })

            del chunks
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"chunk_mb": chunk_mb, "error": str(e)[:200]})

        torch.cuda.empty_cache()
        gc.collect()

    # Calculate efficiency relative to single transfer
    if results and "h2d_bw_gbs" in results[-1]:
        peak_bw = results[-1]["h2d_bw_gbs"]
        for r in results:
            if "h2d_bw_gbs" in r:
                r["efficiency_pct"] = round(r["h2d_bw_gbs"] / max(peak_bw, 0.001) * 100, 1)

    return results


# ============================================================
# Experiment 4: Simulated Alpamayo Layer Prefetch
# ============================================================

def experiment_simulated_layer_prefetch():
    """Alpamayo 레이어 크기로 프리페치 시뮬레이션.
    PyTorch pinned tensor를 사용한다."""
    print("\n" + "=" * 70)
    print("Experiment 4: Simulated Alpamayo Layer Prefetch")
    print("=" * 70)

    # Alpamayo layer sizes (from prior research)
    # NOTE: Expert (4558MB) skipped because pinned memory allocation
    # kills the process under 15GB system RAM constraint.
    layers = [
        ("VLM Layer (single)", 386),  # 0.386 GB
        ("VLM Embedding", 280),
        ("Vision Encoder", 1150),  # 1.15 GB
        ("VLM LM Head", 1280),
        ("Expert (1/4 scale)", 1140),  # 1/4 of 4.56 GB for memory safety
    ]

    results = []

    for name, size_mb in layers:
        n_elements = size_mb * 1024 * 1024 // 2  # FP16
        size_bytes = size_mb * 1024 * 1024
        print(f"\n--- {name} ({size_mb} MB) ---")

        try:
            # Create pinned CPU tensor
            cpu_tensor = torch.randn(n_elements, dtype=torch.float16).pin_memory()

            h2d_times = []
            d2h_times = []

            for trial in range(5):
                torch.cuda.empty_cache()
                gc.collect()

                # H2D
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu_tensor = cpu_tensor.to("cuda", non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                h2d_times.append(start.elapsed_time(end))

                # D2H
                start2 = torch.cuda.Event(enable_timing=True)
                end2 = torch.cuda.Event(enable_timing=True)
                start2.record()
                cpu_back = gpu_tensor.to("cpu", non_blocking=True)
                end2.record()
                torch.cuda.synchronize()
                d2h_times.append(start2.elapsed_time(end2))

                del gpu_tensor, cpu_back

            avg_h2d = np.mean(h2d_times)
            avg_d2h = np.mean(d2h_times)
            h2d_bw = size_bytes / 1e9 / (avg_h2d / 1000) if avg_h2d > 0 else 0
            d2h_bw = size_bytes / 1e9 / (avg_d2h / 1000) if avg_d2h > 0 else 0

            print(f"  H2D: {avg_h2d:.3f} ms ({h2d_bw:.2f} GB/s)")
            print(f"  D2H: {avg_d2h:.3f} ms ({d2h_bw:.2f} GB/s)")

            results.append({
                "name": name,
                "size_mb": size_mb,
                "h2d_ms": round(avg_h2d, 3),
                "d2h_ms": round(avg_d2h, 3),
                "h2d_bw_gbs": round(h2d_bw, 3),
                "d2h_bw_gbs": round(d2h_bw, 3),
                "h2d_times": [round(t, 3) for t in h2d_times],
                "d2h_times": [round(t, 3) for t in d2h_times],
            })

            del cpu_tensor
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"name": name, "size_mb": size_mb, "error": str(e)[:200]})

        torch.cuda.empty_cache()
        gc.collect()

    return results


# ============================================================
# Experiment 5: WSL2 Unified Memory Behavior
# ============================================================

def experiment_unified_memory_wsl2():
    """WSL2에서의 Unified Memory 특이 동작을 분석한다."""
    print("\n" + "=" * 70)
    print("Experiment 5: WSL2 Unified Memory Behavior")
    print("=" * 70)

    results = {}

    # Test 1: Check overcommit behavior
    print("\n[Test 1] CUDA Overcommit behavior...")
    torch.cuda.empty_cache()
    gc.collect()

    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"  Total VRAM: {total_mem / 1e9:.3f} GB")
    print(f"  Free VRAM: {free_mem / 1e9:.3f} GB")

    overcommit = []
    for target_gb in [8, 10, 12, 14, 16, 18, 20]:
        n_elements = int(target_gb * 1e9 / 2)
        t0 = time.time()
        try:
            tensor = torch.randn(n_elements, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            t1 = time.time()

            # Measure access time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = tensor.sum()
            end.record()
            torch.cuda.synchronize()
            access_ms = start.elapsed_time(end)

            entry = {
                "target_gb": target_gb,
                "status": "success",
                "alloc_time_s": round(t1 - t0, 4),
                "access_time_ms": round(access_ms, 3),
                "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
            }
            print(f"  {target_gb}GB: alloc={t1 - t0:.3f}s, access={access_ms:.1f}ms, "
                  f"reported={torch.cuda.memory_allocated(0) / 1e9:.1f}GB")
            overcommit.append(entry)
            del tensor
            torch.cuda.empty_cache()
            gc.collect()
        except RuntimeError as e:
            t1 = time.time()
            entry = {
                "target_gb": target_gb,
                "status": "failed",
                "alloc_time_s": round(t1 - t0, 4),
                "error": str(e)[:200],
            }
            print(f"  {target_gb}GB: FAILED at {t1 - t0:.3f}s")
            overcommit.append(entry)
            torch.cuda.empty_cache()
            gc.collect()

    results["overcommit"] = overcommit

    # Test 2: Access time vs. allocation size (memory pressure)
    print("\n[Test 2] Access time scaling with memory pressure...")
    pressure_results = []

    for pressure_gb in [2, 4, 6, 8, 10, 11]:
        try:
            torch.cuda.empty_cache()
            gc.collect()

            # Allocate background pressure
            n_bg = int(pressure_gb * 1e9 / 2)
            bg_tensor = torch.randn(n_bg, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()

            # Now allocate and access a small tensor (1GB)
            n_test = int(1e9 / 2)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            test_tensor = torch.randn(n_test, device="cuda", dtype=torch.float16)
            end.record()
            torch.cuda.synchronize()
            alloc_ms = start.elapsed_time(end)

            start.record()
            _ = test_tensor.sum()
            end.record()
            torch.cuda.synchronize()
            access_ms = start.elapsed_time(end)

            total_gb = torch.cuda.memory_allocated(0) / 1e9
            exceeds_vram = total_gb > 12.0

            entry = {
                "pressure_gb": pressure_gb,
                "total_allocated_gb": round(total_gb, 3),
                "exceeds_vram": exceeds_vram,
                "test_alloc_ms": round(alloc_ms, 3),
                "test_access_ms": round(access_ms, 3),
            }
            print(f"  {pressure_gb}GB pressure + 1GB test: alloc={alloc_ms:.1f}ms, "
                  f"access={access_ms:.1f}ms (total={total_gb:.1f}GB, exceeds={exceeds_vram})")
            pressure_results.append(entry)

            del bg_tensor, test_tensor
        except Exception as e:
            print(f"  {pressure_gb}GB: FAILED - {str(e)[:100]}")
            pressure_results.append({"pressure_gb": pressure_gb, "error": str(e)[:200]})
            torch.cuda.empty_cache()
            gc.collect()

    results["memory_pressure"] = pressure_results

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(f"Part 2: cudaMemPrefetchAsync 프리페칭 실험")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70)

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    }

    all_results["exp1_prefetch_gpu"] = experiment_prefetch_to_gpu()
    all_results["exp2_prefetch_overlap"] = experiment_prefetch_overlap()
    all_results["exp3_granularity"] = experiment_prefetch_granularity()
    all_results["exp4_layer_prefetch"] = experiment_simulated_layer_prefetch()
    all_results["exp5_unified_wsl2"] = experiment_unified_memory_wsl2()

    out_path = os.path.join(OUTPUT_DIR, "prefetch_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    results = main()
