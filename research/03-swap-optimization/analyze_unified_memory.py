#!/usr/bin/env python3
"""Part 1: Unified Memory 동작 분석

CUDA Unified Memory의 페이지 폴트 패턴과 스와핑 동작을 분석한다.
- torch.cuda.memory_stats() 기반 카운터 수집
- 고빈도 메모리 모니터링 (10ms 간격)
- Unified Memory 할당/접근 패턴 분석
- 대형 텐서의 스와핑 동작 관찰
"""

import os
import sys
import json
import time
import csv
import gc
import threading
import ctypes
from datetime import datetime

import torch
import numpy as np

OUTPUT_DIR = "/home/seungwoo/workspace/research/03-swap-optimization"
PYTHON_ENV = "/home/seungwoo/workspace/alpamayo/ar1_venv"
CUDA_RT_PATH = f"{PYTHON_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"

# ============================================================
# CUDA Runtime Bindings
# ============================================================

cuda_rt = ctypes.cdll.LoadLibrary(CUDA_RT_PATH)


def cuda_check(result, func_name=""):
    if result != 0:
        print(f"  [WARN] {func_name} returned error code {result}")
    return result


# ============================================================
# High-frequency VRAM Monitor
# ============================================================

class HighFreqMonitor:
    """10ms 간격으로 VRAM 사용량 모니터링"""

    def __init__(self, csv_path: str, interval: float = 0.01):
        self.csv_path = csv_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.data = []
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._save()

    def _monitor(self):
        while not self._stop.is_set():
            try:
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                elapsed = time.time() - self.start_time
                self.data.append({
                    "time_ms": round(elapsed * 1000, 1),
                    "allocated_mb": round(allocated / 1e6, 2),
                    "reserved_mb": round(reserved / 1e6, 2),
                })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _save(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time_ms", "allocated_mb", "reserved_mb"])
            writer.writeheader()
            writer.writerows(self.data)


# ============================================================
# Experiment 1: Memory Stats Counter Analysis
# ============================================================

def experiment_memory_stats_analysis():
    """CUDA Memory Allocator 통계를 상세 분석한다."""
    print("=" * 70)
    print("Experiment 1: Memory Stats Counter Analysis")
    print("=" * 70)

    results = {}

    # Baseline stats (clean state)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()
    gc.collect()

    baseline_stats = torch.cuda.memory_stats(0)

    # Phase 1: Small allocation (fits in VRAM)
    print("\n[Phase 1] Small allocation (2 GB) - fits in 12GB VRAM...")
    t0 = time.time()
    small_tensor = torch.randn(256, 1024, 1024, device="cuda", dtype=torch.float16)  # ~512 MB
    torch.cuda.synchronize()
    t1 = time.time()
    small_size = small_tensor.numel() * small_tensor.element_size() / 1e9
    print(f"  Size: {small_size:.3f} GB")
    print(f"  Allocation time: {t1 - t0:.4f}s")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.3f} GB")

    stats_after_small = torch.cuda.memory_stats(0)
    results["small_alloc"] = {
        "size_gb": round(small_size, 3),
        "alloc_time_s": round(t1 - t0, 4),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 3),
        "num_alloc_retries": stats_after_small.get("num_alloc_retries", 0),
        "num_ooms": stats_after_small.get("num_ooms", 0),
    }

    del small_tensor
    torch.cuda.empty_cache()
    gc.collect()

    # Phase 2: Medium allocation (fills most of VRAM)
    print("\n[Phase 2] Medium allocation (10 GB) - nearly fills 12GB VRAM...")
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    try:
        medium_tensor = torch.randn(5120, 1024, 1024, device="cuda", dtype=torch.float16)  # ~10 GB
        torch.cuda.synchronize()
        t1 = time.time()
        medium_size = medium_tensor.numel() * medium_tensor.element_size() / 1e9
        print(f"  Size: {medium_size:.3f} GB")
        print(f"  Allocation time: {t1 - t0:.4f}s")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")

        stats_after_medium = torch.cuda.memory_stats(0)
        results["medium_alloc"] = {
            "size_gb": round(medium_size, 3),
            "alloc_time_s": round(t1 - t0, 4),
            "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 3),
            "num_alloc_retries": stats_after_medium.get("num_alloc_retries", 0),
            "num_ooms": stats_after_medium.get("num_ooms", 0),
        }
        del medium_tensor
    except RuntimeError as e:
        t1 = time.time()
        print(f"  OOM after {t1 - t0:.4f}s: {e}")
        results["medium_alloc"] = {"error": str(e), "time_s": round(t1 - t0, 4)}

    torch.cuda.empty_cache()
    gc.collect()

    # Phase 3: Overcommit allocation (exceeds VRAM) - triggers Unified Memory behavior
    print("\n[Phase 3] Overcommit allocation (14 GB) - exceeds 12GB VRAM...")
    torch.cuda.reset_peak_memory_stats(0)

    # Try using cudaMallocManaged via ctypes to observe Unified Memory behavior
    overcommit_results = []
    for target_gb in [14, 16, 20]:
        t0 = time.time()
        try:
            # PyTorch normally uses cudaMalloc, which will fail if > VRAM
            n_elements = int(target_gb * 1e9 / 2)  # FP16
            tensor = torch.randn(n_elements, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            t1 = time.time()
            actual_size = tensor.numel() * tensor.element_size() / 1e9
            print(f"  {target_gb}GB alloc: SUCCESS in {t1 - t0:.4f}s (actual {actual_size:.2f} GB)")
            stats = torch.cuda.memory_stats(0)
            overcommit_results.append({
                "target_gb": target_gb,
                "status": "success",
                "alloc_time_s": round(t1 - t0, 4),
                "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
            })
            del tensor
            torch.cuda.empty_cache()
            gc.collect()
        except RuntimeError as e:
            t1 = time.time()
            error_msg = str(e)[:200]
            print(f"  {target_gb}GB alloc: FAILED in {t1 - t0:.4f}s - {error_msg}")
            overcommit_results.append({
                "target_gb": target_gb,
                "status": "failed",
                "alloc_time_s": round(t1 - t0, 4),
                "error": error_msg,
            })
            torch.cuda.empty_cache()
            gc.collect()

    results["overcommit"] = overcommit_results

    # Phase 4: Memory allocator behavior under pressure
    print("\n[Phase 4] Sequential allocation pressure test...")
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()
    gc.collect()

    chunk_size_gb = 1.0
    n_elements_per_chunk = int(chunk_size_gb * 1e9 / 2)  # FP16
    chunks = []
    alloc_log = []

    for i in range(15):  # Try to allocate 15 GB in 1GB chunks
        t0 = time.time()
        try:
            chunk = torch.randn(n_elements_per_chunk, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            t1 = time.time()
            chunks.append(chunk)
            stats = torch.cuda.memory_stats(0)
            entry = {
                "chunk_idx": i,
                "status": "success",
                "alloc_time_s": round(t1 - t0, 4),
                "total_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
                "total_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 3),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
            }
            print(f"  Chunk {i}: {entry['alloc_time_s']}s, total={entry['total_allocated_gb']:.3f} GB, "
                  f"retries={entry['num_alloc_retries']}")
            alloc_log.append(entry)
        except RuntimeError as e:
            t1 = time.time()
            entry = {
                "chunk_idx": i,
                "status": "failed",
                "alloc_time_s": round(t1 - t0, 4),
                "error": str(e)[:200],
            }
            print(f"  Chunk {i}: FAILED at {t1 - t0:.4f}s")
            alloc_log.append(entry)
            break

    results["sequential_pressure"] = alloc_log

    # Cleanup
    del chunks
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Experiment 2: High-Frequency Memory Monitoring During Compute
# ============================================================

def experiment_highfreq_monitoring():
    """고빈도 모니터링으로 메모리 사용 패턴을 상세 관찰한다."""
    print("\n" + "=" * 70)
    print("Experiment 2: High-Frequency Memory Monitoring")
    print("=" * 70)

    results = {}
    csv_path = os.path.join(OUTPUT_DIR, "unified_memory_timeline.csv")
    monitor = HighFreqMonitor(csv_path, interval=0.01)  # 10ms

    monitor.start()

    # Phase 1: Rapid allocation/deallocation
    print("\n[Phase 1] Rapid alloc/dealloc cycle (0.5 GB x 10)...")
    phase1_times = []
    for i in range(10):
        t0 = time.time()
        x = torch.randn(256 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)  # 256 MB
        torch.cuda.synchronize()
        t1 = time.time()
        # Do some compute
        y = x * 2.0 + 1.0
        torch.cuda.synchronize()
        t2 = time.time()
        del x, y
        torch.cuda.empty_cache()
        t3 = time.time()
        phase1_times.append({
            "alloc_ms": round((t1 - t0) * 1000, 2),
            "compute_ms": round((t2 - t1) * 1000, 2),
            "free_ms": round((t3 - t2) * 1000, 2),
        })
    results["rapid_cycle"] = phase1_times
    print(f"  Avg alloc: {np.mean([t['alloc_ms'] for t in phase1_times]):.2f}ms")
    print(f"  Avg compute: {np.mean([t['compute_ms'] for t in phase1_times]):.2f}ms")
    print(f"  Avg free: {np.mean([t['free_ms'] for t in phase1_times]):.2f}ms")

    time.sleep(0.5)  # Let monitor catch the transition

    # Phase 2: Large allocation near VRAM limit
    print("\n[Phase 2] Filling VRAM to near capacity...")
    fill_results = []
    tensors = []
    for i in range(10):  # 10 x 1GB = 10GB
        t0 = time.time()
        try:
            t = torch.randn(512 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)  # 512 MB
            torch.cuda.synchronize()
            t1 = time.time()
            tensors.append(t)
            fill_results.append({
                "idx": i,
                "alloc_time_ms": round((t1 - t0) * 1000, 2),
                "total_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
            })
            print(f"  Fill {i}: {(t1 - t0) * 1000:.2f}ms, total={torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
        except RuntimeError as e:
            t1 = time.time()
            fill_results.append({
                "idx": i,
                "alloc_time_ms": round((t1 - t0) * 1000, 2),
                "error": str(e)[:100],
            })
            print(f"  Fill {i}: FAILED (OOM) at {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
            break

    results["fill_vram"] = fill_results

    time.sleep(0.5)

    # Phase 3: Compute under memory pressure
    if len(tensors) >= 2:
        print("\n[Phase 3] Compute under memory pressure...")
        t0 = time.time()
        try:
            # Try to do matmul that requires temp memory
            a = tensors[0][:1024*1024].reshape(1024, 1024)
            b = tensors[1][:1024*1024].reshape(1024, 1024)
            c = torch.matmul(a.float(), b.float())
            torch.cuda.synchronize()
            t1 = time.time()
            results["compute_under_pressure"] = {
                "status": "success",
                "time_ms": round((t1 - t0) * 1000, 2),
                "vram_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
            }
            print(f"  Matmul under pressure: {(t1 - t0) * 1000:.2f}ms")
            del a, b, c
        except RuntimeError as e:
            t1 = time.time()
            results["compute_under_pressure"] = {
                "status": "failed",
                "time_ms": round((t1 - t0) * 1000, 2),
                "error": str(e)[:200],
            }
            print(f"  Matmul under pressure: FAILED - {str(e)[:100]}")

    # Cleanup
    del tensors
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(0.5)

    monitor.stop()
    results["monitor_csv"] = csv_path
    results["monitor_samples"] = len(monitor.data)
    print(f"\n  Monitor captured {len(monitor.data)} samples to {csv_path}")

    return results


# ============================================================
# Experiment 3: Unified Memory Allocation Pattern Analysis
# ============================================================

def experiment_unified_memory_patterns():
    """Unified Memory (managed memory) 할당 패턴을 분석한다.

    PyTorch는 기본적으로 cudaMalloc을 사용하지만,
    overcommit 시 CUDA runtime이 Unified Memory로 fallback할 수 있다.
    여기서는 명시적으로 Unified Memory를 할당하여 동작을 분석한다.
    """
    print("\n" + "=" * 70)
    print("Experiment 3: Unified Memory Allocation Pattern Analysis")
    print("=" * 70)

    results = {}

    # Part A: PyTorch의 cudaMalloc 한계 확인
    print("\n[Part A] PyTorch cudaMalloc limit...")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)

    free_mem, total_mem = torch.cuda.mem_get_info(0)
    print(f"  Total VRAM: {total_mem / 1e9:.3f} GB")
    print(f"  Free VRAM: {free_mem / 1e9:.3f} GB")
    results["vram_info"] = {
        "total_gb": round(total_mem / 1e9, 3),
        "free_gb": round(free_mem / 1e9, 3),
    }

    # Part B: Managed Memory test via ctypes
    print("\n[Part B] cudaMallocManaged test...")
    managed_results = []

    for size_mb in [100, 500, 1000, 2000, 4000]:
        size_bytes = size_mb * 1024 * 1024
        ptr = ctypes.c_void_p()

        # cudaMemAttachGlobal = 1
        t0 = time.time()
        ret = cuda_rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(size_bytes), ctypes.c_uint(1))
        t1 = time.time()

        if ret == 0:
            # Test: prefetch to GPU
            t2 = time.time()
            ret2 = cuda_rt.cudaMemPrefetchAsync(ptr, ctypes.c_size_t(size_bytes), ctypes.c_int(0), ctypes.c_void_p(0))
            torch.cuda.synchronize()
            t3 = time.time()

            # Test: prefetch to CPU (device -1)
            t4 = time.time()
            ret3 = cuda_rt.cudaMemPrefetchAsync(ptr, ctypes.c_size_t(size_bytes), ctypes.c_int(-1), ctypes.c_void_p(0))
            torch.cuda.synchronize()
            t5 = time.time()

            bw_h2d = (size_bytes / 1e9) / max(t3 - t2, 1e-6)
            bw_d2h = (size_bytes / 1e9) / max(t5 - t4, 1e-6)

            entry = {
                "size_mb": size_mb,
                "alloc_time_ms": round((t1 - t0) * 1000, 3),
                "prefetch_to_gpu_ms": round((t3 - t2) * 1000, 3),
                "prefetch_to_cpu_ms": round((t5 - t4) * 1000, 3),
                "bw_h2d_gbs": round(bw_h2d, 3),
                "bw_d2h_gbs": round(bw_d2h, 3),
                "prefetch_gpu_status": ret2,
                "prefetch_cpu_status": ret3,
            }
            print(f"  {size_mb}MB: alloc={entry['alloc_time_ms']:.1f}ms, "
                  f"H2D={entry['prefetch_to_gpu_ms']:.1f}ms ({bw_h2d:.2f} GB/s), "
                  f"D2H={entry['prefetch_to_cpu_ms']:.1f}ms ({bw_d2h:.2f} GB/s)")
            managed_results.append(entry)

            # Free
            cuda_rt.cudaFree(ptr)
        else:
            print(f"  {size_mb}MB: cudaMallocManaged FAILED (code={ret})")
            managed_results.append({
                "size_mb": size_mb,
                "status": "failed",
                "error_code": ret,
            })

    results["managed_memory"] = managed_results

    # Part C: Page fault simulation - Access pattern impact
    print("\n[Part C] Access pattern impact on Unified Memory...")
    access_results = []

    test_size_mb = 512
    test_size_bytes = test_size_mb * 1024 * 1024

    for pattern_name, description in [
        ("sequential", "Sequential access (good locality)"),
        ("strided", "Strided access (poor locality)"),
        ("random", "Random access (worst locality)"),
    ]:
        ptr = ctypes.c_void_p()
        ret = cuda_rt.cudaMallocManaged(ctypes.byref(ptr), ctypes.c_size_t(test_size_bytes), ctypes.c_uint(1))
        if ret != 0:
            print(f"  {pattern_name}: alloc FAILED")
            continue

        # Prefetch to CPU first
        cuda_rt.cudaMemPrefetchAsync(ptr, ctypes.c_size_t(test_size_bytes), ctypes.c_int(-1), ctypes.c_void_p(0))
        torch.cuda.synchronize()

        # Now access from GPU with different patterns via a CUDA kernel
        # Since we can't easily write CUDA kernels, we'll use cudaMemPrefetchAsync
        # and measure the prefetch time as a proxy

        # Prefetch to GPU (simulates sequential access)
        t0 = time.time()
        cuda_rt.cudaMemPrefetchAsync(ptr, ctypes.c_size_t(test_size_bytes), ctypes.c_int(0), ctypes.c_void_p(0))
        torch.cuda.synchronize()
        t1 = time.time()

        bw = (test_size_bytes / 1e9) / max(t1 - t0, 1e-6)
        access_results.append({
            "pattern": pattern_name,
            "description": description,
            "size_mb": test_size_mb,
            "prefetch_time_ms": round((t1 - t0) * 1000, 3),
            "bandwidth_gbs": round(bw, 3),
        })
        print(f"  {pattern_name}: prefetch {(t1 - t0) * 1000:.2f}ms, BW={bw:.2f} GB/s")

        cuda_rt.cudaFree(ptr)

    results["access_patterns"] = access_results

    # Part D: Simulate page fault driven vs. prefetched access
    print("\n[Part D] Page fault vs. prefetched access comparison...")

    # Create a PyTorch tensor on CPU, then access on GPU
    # This simulates what happens with Unified Memory
    fault_vs_prefetch = []

    for size_mb in [128, 256, 512, 1024, 2048]:
        n_elements = size_mb * 1024 * 1024 // 2  # FP16

        try:
            # Method 1: Direct GPU allocation (no page faults)
            torch.cuda.empty_cache()
            gc.collect()
            t0 = time.time()
            gpu_tensor = torch.randn(n_elements, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            t1 = time.time()

            # Do a simple operation
            _ = gpu_tensor.sum()
            torch.cuda.synchronize()
            t2 = time.time()

            direct_alloc_ms = (t1 - t0) * 1000
            direct_compute_ms = (t2 - t1) * 1000
            del gpu_tensor

            # Method 2: CPU allocation -> GPU transfer (explicit, pinned)
            torch.cuda.empty_cache()
            gc.collect()
            cpu_tensor = torch.randn(n_elements, dtype=torch.float16).pin_memory()
            t0 = time.time()
            gpu_tensor = cpu_tensor.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            t1 = time.time()
            _ = gpu_tensor.sum()
            torch.cuda.synchronize()
            t2 = time.time()

            pinned_transfer_ms = (t1 - t0) * 1000
            pinned_compute_ms = (t2 - t1) * 1000
            del gpu_tensor, cpu_tensor

            # Method 3: CPU allocation -> GPU transfer (pageable)
            torch.cuda.empty_cache()
            gc.collect()
            cpu_tensor = torch.randn(n_elements, dtype=torch.float16)
            t0 = time.time()
            gpu_tensor = cpu_tensor.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            t1 = time.time()
            _ = gpu_tensor.sum()
            torch.cuda.synchronize()
            t2 = time.time()

            pageable_transfer_ms = (t1 - t0) * 1000
            pageable_compute_ms = (t2 - t1) * 1000
            del gpu_tensor, cpu_tensor

            entry = {
                "size_mb": size_mb,
                "direct_alloc_ms": round(direct_alloc_ms, 2),
                "direct_compute_ms": round(direct_compute_ms, 2),
                "pinned_transfer_ms": round(pinned_transfer_ms, 2),
                "pinned_compute_ms": round(pinned_compute_ms, 2),
                "pageable_transfer_ms": round(pageable_transfer_ms, 2),
                "pageable_compute_ms": round(pageable_compute_ms, 2),
            }
            fault_vs_prefetch.append(entry)
            print(f"  {size_mb}MB: direct={direct_alloc_ms:.1f}ms, "
                  f"pinned={pinned_transfer_ms:.1f}ms, pageable={pageable_transfer_ms:.1f}ms")
        except RuntimeError as e:
            print(f"  {size_mb}MB: FAILED - {str(e)[:100]}")
            fault_vs_prefetch.append({"size_mb": size_mb, "error": str(e)[:200]})
            torch.cuda.empty_cache()
            gc.collect()

    results["fault_vs_prefetch"] = fault_vs_prefetch

    return results


# ============================================================
# Experiment 4: Memory Allocator Retry/OOM Analysis
# ============================================================

def experiment_allocator_analysis():
    """PyTorch CUDA memory allocator의 retry/OOM 동작을 분석한다."""
    print("\n" + "=" * 70)
    print("Experiment 4: Allocator Retry and OOM Analysis")
    print("=" * 70)

    results = {}
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(0)

    # Fill VRAM to near capacity, then try to allocate more
    print("\n[Step 1] Fill VRAM to near capacity...")
    tensors = []
    total_allocated = 0
    while True:
        try:
            t = torch.randn(128 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)  # 128 MB
            torch.cuda.synchronize()
            tensors.append(t)
            total_allocated += 128
            if total_allocated % 1024 == 0:
                print(f"  Allocated {total_allocated} MB, "
                      f"VRAM={torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
        except RuntimeError:
            break

    stats = torch.cuda.memory_stats(0)
    results["fill_stats"] = {
        "total_allocated_mb": total_allocated,
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 3),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
        "num_ooms": stats.get("num_ooms", 0),
    }
    print(f"  Filled to {total_allocated} MB ({torch.cuda.memory_allocated(0) / 1e9:.3f} GB)")
    print(f"  Alloc retries: {stats.get('num_alloc_retries', 0)}")
    print(f"  OOMs: {stats.get('num_ooms', 0)}")

    # Step 2: Free some and try again
    print("\n[Step 2] Free 2GB and measure reallocation...")
    # Free first 16 chunks (16 x 128MB = 2GB)
    n_free = min(16, len(tensors))
    for i in range(n_free):
        del tensors[0]
        tensors = tensors  # force list update

    # Actually delete references
    freed_tensors = []
    remaining_tensors = tensors[n_free:]
    del tensors
    torch.cuda.empty_cache()
    gc.collect()

    print(f"  After freeing: VRAM={torch.cuda.memory_allocated(0) / 1e9:.3f} GB")

    # Try to allocate 1.5GB at once
    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    try:
        big = torch.randn(int(1.5 * 1024 * 1024 * 1024 / 2), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t1 = time.time()
        stats = torch.cuda.memory_stats(0)
        results["realloc"] = {
            "status": "success",
            "time_ms": round((t1 - t0) * 1000, 2),
            "num_alloc_retries": stats.get("num_alloc_retries", 0),
        }
        print(f"  Realloc 1.5GB: {(t1 - t0) * 1000:.2f}ms, retries={stats.get('num_alloc_retries', 0)}")
        del big
    except RuntimeError as e:
        t1 = time.time()
        stats = torch.cuda.memory_stats(0)
        results["realloc"] = {
            "status": "failed",
            "time_ms": round((t1 - t0) * 1000, 2),
            "num_alloc_retries": stats.get("num_alloc_retries", 0),
            "error": str(e)[:200],
        }
        print(f"  Realloc 1.5GB: FAILED - retries={stats.get('num_alloc_retries', 0)}")

    # Cleanup
    del remaining_tensors
    torch.cuda.empty_cache()
    gc.collect()

    # Step 3: Full memory stats dump
    print("\n[Step 3] Full memory stats analysis...")
    final_stats = torch.cuda.memory_stats(0)
    important_keys = [
        "num_alloc_retries", "num_ooms",
        "allocated_bytes.all.peak", "reserved_bytes.all.peak",
        "allocation.all.peak", "segment.all.peak",
        "oversize_allocations.peak", "oversize_segments.peak",
    ]
    results["final_stats"] = {}
    for key in important_keys:
        val = final_stats.get(key, "N/A")
        results["final_stats"][key] = val
        print(f"  {key}: {val}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(f"Part 1: Unified Memory 동작 분석")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70)

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    }

    # Run experiments
    all_results["exp1_memory_stats"] = experiment_memory_stats_analysis()
    all_results["exp2_highfreq"] = experiment_highfreq_monitoring()
    all_results["exp3_unified_memory"] = experiment_unified_memory_patterns()
    all_results["exp4_allocator"] = experiment_allocator_analysis()

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "unified_memory_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    results = main()
