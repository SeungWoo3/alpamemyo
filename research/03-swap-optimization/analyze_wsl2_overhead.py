#!/usr/bin/env python3
"""Part 4: WSL2 오버헤드 분석

WSL2 환경에서의 메모리 전송 오버헤드를 측정한다:
1. Pageable vs Pinned memory 전송 속도 비교
2. 단일 대형 전송 vs 다수 소형 전송 비교
3. CUDA 이벤트 기반 정밀 시간 측정
4. WSL2 가상화 계층의 영향 분석

NOTE: 시스템 RAM 15GB 제약으로 pinned memory 크기 제한.
NOTE: WSL2에서 cudaMemPrefetchAsync(cpu) 미지원 확인됨.
"""

import os
import sys
import json
import time
import gc
import subprocess
from datetime import datetime

import torch
import numpy as np

OUTPUT_DIR = "/home/seungwoo/workspace/research/03-swap-optimization"


# ============================================================
# Helper Functions
# ============================================================

def get_system_info():
    """시스템 정보 수집"""
    info = {}
    try:
        r = subprocess.run(["uname", "-r"], capture_output=True, text=True)
        info["kernel"] = r.stdout.strip()
    except:
        info["kernel"] = "unknown"

    try:
        r = subprocess.run(["free", "-m"], capture_output=True, text=True)
        info["memory_info"] = r.stdout.strip()
    except:
        pass

    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
                           "--format=csv,noheader"], capture_output=True, text=True)
        info["gpu_pcie"] = r.stdout.strip()
    except:
        info["gpu_pcie"] = "nvidia-smi query failed"

    return info


# ============================================================
# Experiment 1: Pageable vs Pinned Memory Overhead
# ============================================================

def experiment_pageable_vs_pinned():
    """Pageable vs Pinned memory의 전송 오버헤드를 정밀 측정한다."""
    print("=" * 70)
    print("Experiment 1: Pageable vs Pinned Memory Transfer Overhead")
    print("=" * 70)

    results = []
    n_warmup = 3
    n_trials = 10

    # Reduced max size to 512MB for memory safety
    for size_mb in [1, 4, 16, 64, 128, 256, 512]:
        n_elements = size_mb * 1024 * 1024 // 2  # FP16
        size_bytes = size_mb * 1024 * 1024

        print(f"\n--- {size_mb} MB ---")
        entry = {"size_mb": size_mb, "size_bytes": size_bytes}

        # ---- Pageable Memory ----
        try:
            cpu_pageable = torch.randn(n_elements, dtype=torch.float16)

            # H2D
            h2d_times = []
            for i in range(n_warmup + n_trials):
                torch.cuda.empty_cache()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu = cpu_pageable.to("cuda")
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    h2d_times.append(start.elapsed_time(end))
                del gpu

            entry["pageable_h2d_ms"] = round(np.mean(h2d_times), 3)
            entry["pageable_h2d_std"] = round(np.std(h2d_times), 3)
            entry["pageable_h2d_bw"] = round(size_bytes / 1e9 / (np.mean(h2d_times) / 1000), 3)

            # D2H
            gpu = cpu_pageable.to("cuda")
            torch.cuda.synchronize()
            d2h_times = []
            for i in range(n_warmup + n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cpu_back = gpu.to("cpu")
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    d2h_times.append(start.elapsed_time(end))
                del cpu_back

            entry["pageable_d2h_ms"] = round(np.mean(d2h_times), 3)
            entry["pageable_d2h_std"] = round(np.std(d2h_times), 3)
            entry["pageable_d2h_bw"] = round(size_bytes / 1e9 / (np.mean(d2h_times) / 1000), 3)
            del gpu, cpu_pageable
        except Exception as e:
            entry["pageable_error"] = str(e)[:200]

        torch.cuda.empty_cache()
        gc.collect()

        # ---- Pinned Memory ----
        try:
            cpu_pinned = torch.randn(n_elements, dtype=torch.float16).pin_memory()

            # H2D
            h2d_times = []
            for i in range(n_warmup + n_trials):
                torch.cuda.empty_cache()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu = cpu_pinned.to("cuda", non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    h2d_times.append(start.elapsed_time(end))
                del gpu

            entry["pinned_h2d_ms"] = round(np.mean(h2d_times), 3)
            entry["pinned_h2d_std"] = round(np.std(h2d_times), 3)
            entry["pinned_h2d_bw"] = round(size_bytes / 1e9 / (np.mean(h2d_times) / 1000), 3)

            # D2H (pinned destination)
            gpu = cpu_pinned.to("cuda")
            torch.cuda.synchronize()
            d2h_times = []
            for i in range(n_warmup + n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                cpu_dst = torch.empty_like(cpu_pinned).pin_memory()
                start.record()
                cpu_dst.copy_(gpu, non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    d2h_times.append(start.elapsed_time(end))
                del cpu_dst

            entry["pinned_d2h_ms"] = round(np.mean(d2h_times), 3)
            entry["pinned_d2h_std"] = round(np.std(d2h_times), 3)
            entry["pinned_d2h_bw"] = round(size_bytes / 1e9 / (np.mean(d2h_times) / 1000), 3)
            del gpu, cpu_pinned
        except Exception as e:
            entry["pinned_error"] = str(e)[:200]

        torch.cuda.empty_cache()
        gc.collect()

        # ---- Speedup ----
        if "pinned_h2d_bw" in entry and "pageable_h2d_bw" in entry:
            entry["h2d_pinned_speedup"] = round(entry["pinned_h2d_bw"] / max(entry["pageable_h2d_bw"], 0.001), 2)
        if "pinned_d2h_bw" in entry and "pageable_d2h_bw" in entry:
            entry["d2h_pinned_speedup"] = round(entry["pinned_d2h_bw"] / max(entry["pageable_d2h_bw"], 0.001), 2)

        print(f"  Pageable H2D: {entry.get('pageable_h2d_ms', 'N/A')} ms ({entry.get('pageable_h2d_bw', 'N/A')} GB/s)")
        print(f"  Pinned  H2D: {entry.get('pinned_h2d_ms', 'N/A')} ms ({entry.get('pinned_h2d_bw', 'N/A')} GB/s)")
        print(f"  Pageable D2H: {entry.get('pageable_d2h_ms', 'N/A')} ms ({entry.get('pageable_d2h_bw', 'N/A')} GB/s)")
        print(f"  Pinned  D2H: {entry.get('pinned_d2h_ms', 'N/A')} ms ({entry.get('pinned_d2h_bw', 'N/A')} GB/s)")
        if "h2d_pinned_speedup" in entry:
            print(f"  H2D speedup: {entry['h2d_pinned_speedup']}x, D2H speedup: {entry.get('d2h_pinned_speedup', 'N/A')}x")

        results.append(entry)

    return results


# ============================================================
# Experiment 2: Single Large vs Many Small Transfers
# ============================================================

def experiment_large_vs_small():
    """단일 대형 전송 vs 다수 소형 전송을 비교한다."""
    print("\n" + "=" * 70)
    print("Experiment 2: Single Large vs Many Small Transfers")
    print("=" * 70)

    results = []
    total_size_mb = 256  # Reduced from 512MB for memory safety
    total_bytes = total_size_mb * 1024 * 1024
    n_trials = 5

    chunk_sizes_mb = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

    for chunk_mb in chunk_sizes_mb:
        chunk_bytes = int(chunk_mb * 1024 * 1024)
        n_chunks = total_bytes // chunk_bytes
        chunk_elements = chunk_bytes // 2  # FP16

        print(f"\n--- {total_size_mb}MB as {n_chunks} x {chunk_mb}MB chunks ---")

        try:
            # Create pinned CPU chunks one at a time to reduce peak RAM
            h2d_times = []
            for trial in range(n_trials):
                torch.cuda.empty_cache()
                gc.collect()

                chunks = []
                for _ in range(n_chunks):
                    chunks.append(torch.randn(chunk_elements, dtype=torch.float16).pin_memory())

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
                del gpu_chunks, chunks
                torch.cuda.empty_cache()
                gc.collect()

            avg_h2d = np.mean(h2d_times)
            h2d_bw = total_bytes / 1e9 / (avg_h2d / 1000)

            print(f"  H2D: {avg_h2d:.3f}ms ({h2d_bw:.2f} GB/s)")

            entry = {
                "chunk_mb": chunk_mb,
                "n_chunks": n_chunks,
                "h2d_ms": round(avg_h2d, 3),
                "h2d_bw_gbs": round(h2d_bw, 3),
            }
            results.append(entry)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"chunk_mb": chunk_mb, "error": str(e)[:200]})

        torch.cuda.empty_cache()
        gc.collect()

    # Calculate overhead ratios relative to single transfer (largest chunk)
    if results and "h2d_ms" in results[-1]:
        base_h2d = results[-1]["h2d_ms"]
        for r in results:
            if "h2d_ms" in r:
                r["h2d_overhead_ratio"] = round(r["h2d_ms"] / base_h2d, 3)

    return results


# ============================================================
# Experiment 3: Memory Transfer Latency Analysis
# ============================================================

def experiment_transfer_latency():
    """매우 작은 전송의 고정 오버헤드(latency)를 측정한다."""
    print("\n" + "=" * 70)
    print("Experiment 3: Transfer Latency Analysis (Fixed Overhead)")
    print("=" * 70)

    results = []
    n_trials = 20

    for size_bytes in [4, 16, 64, 256, 1024, 4096, 16384, 65536,
                       256 * 1024, 1024 * 1024, 4 * 1024 * 1024,
                       16 * 1024 * 1024, 64 * 1024 * 1024]:
        n_elements = max(1, size_bytes // 2)
        actual_bytes = n_elements * 2

        try:
            cpu_tensor = torch.randn(n_elements, dtype=torch.float16).pin_memory()

            h2d_times = []
            for i in range(n_trials):
                torch.cuda.empty_cache()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu = cpu_tensor.to("cuda", non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                h2d_times.append(start.elapsed_time(end))
                del gpu

            avg_h2d = np.mean(h2d_times)
            min_h2d = np.min(h2d_times)
            bw = actual_bytes / 1e9 / (avg_h2d / 1000) if avg_h2d > 0 else 0

            size_str = f"{actual_bytes}" if actual_bytes < 1024 else \
                       f"{actual_bytes / 1024:.0f}KB" if actual_bytes < 1024 * 1024 else \
                       f"{actual_bytes / (1024 * 1024):.0f}MB"

            print(f"  {size_str:>8s}: avg={avg_h2d:.3f}ms, min={min_h2d:.3f}ms, BW={bw:.2f} GB/s")

            results.append({
                "size_bytes": actual_bytes,
                "size_label": size_str,
                "h2d_avg_ms": round(avg_h2d, 4),
                "h2d_min_ms": round(min_h2d, 4),
                "h2d_std_ms": round(np.std(h2d_times), 4),
                "bandwidth_gbs": round(bw, 3),
            })

            del cpu_tensor
        except Exception as e:
            print(f"  {size_bytes}: FAILED - {e}")

        torch.cuda.empty_cache()
        gc.collect()

    # Estimate fixed overhead (latency) from smallest transfers
    if len(results) >= 2:
        fixed_overhead_ms = results[0]["h2d_min_ms"]
        print(f"\n  Estimated fixed overhead (latency): {fixed_overhead_ms:.4f} ms")

    return results


# ============================================================
# Experiment 4: Bidirectional Transfer Contention
# ============================================================

def experiment_bidirectional_contention():
    """양방향 동시 전송의 경합 효과를 측정한다."""
    print("\n" + "=" * 70)
    print("Experiment 4: Bidirectional Transfer Contention")
    print("=" * 70)

    size_mb = 256  # Reduced for memory safety
    n_elements = size_mb * 1024 * 1024 // 2
    size_bytes = size_mb * 1024 * 1024
    n_trials = 5

    print(f"\nTest size: {size_mb} MB")

    results = {}

    try:
        # Test 1: H2D only
        print("\n[Test 1] H2D only...")
        cpu_tensor = torch.randn(n_elements, dtype=torch.float16).pin_memory()
        h2d_only_times = []
        for i in range(n_trials):
            torch.cuda.empty_cache()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gpu = cpu_tensor.to("cuda", non_blocking=True)
            end.record()
            torch.cuda.synchronize()
            h2d_only_times.append(start.elapsed_time(end))
            del gpu
        avg_h2d = np.mean(h2d_only_times)
        h2d_bw = size_bytes / 1e9 / (avg_h2d / 1000)
        print(f"  H2D: {avg_h2d:.3f} ms ({h2d_bw:.2f} GB/s)")

        # Test 2: D2H only
        print("\n[Test 2] D2H only...")
        gpu_tensor = cpu_tensor.to("cuda")
        torch.cuda.synchronize()
        d2h_only_times = []
        for i in range(n_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            cpu_dst = torch.empty_like(cpu_tensor).pin_memory()
            start.record()
            cpu_dst.copy_(gpu_tensor, non_blocking=True)
            end.record()
            torch.cuda.synchronize()
            d2h_only_times.append(start.elapsed_time(end))
            del cpu_dst
        avg_d2h = np.mean(d2h_only_times)
        d2h_bw = size_bytes / 1e9 / (avg_d2h / 1000)
        print(f"  D2H: {avg_d2h:.3f} ms ({d2h_bw:.2f} GB/s)")
        del gpu_tensor

        # Test 3: Bidirectional simultaneous
        print("\n[Test 3] Bidirectional simultaneous (H2D + D2H)...")

        cpu_h2d = torch.randn(n_elements, dtype=torch.float16).pin_memory()
        gpu_d2h = torch.randn(n_elements, device="cuda", dtype=torch.float16)
        cpu_d2h_dst = torch.empty(n_elements, dtype=torch.float16).pin_memory()
        torch.cuda.synchronize()

        stream_h2d = torch.cuda.Stream()
        stream_d2h = torch.cuda.Stream()

        bidir_times = []
        for i in range(n_trials):
            start = torch.cuda.Event(enable_timing=True)
            end_h2d = torch.cuda.Event(enable_timing=True)
            end_d2h = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)

            start.record()

            with torch.cuda.stream(stream_h2d):
                gpu_result = cpu_h2d.to("cuda", non_blocking=True)
                end_h2d.record(stream_h2d)

            with torch.cuda.stream(stream_d2h):
                cpu_d2h_dst.copy_(gpu_d2h, non_blocking=True)
                end_d2h.record(stream_d2h)

            torch.cuda.current_stream().wait_stream(stream_h2d)
            torch.cuda.current_stream().wait_stream(stream_d2h)
            end_total.record()
            torch.cuda.synchronize()

            total_ms = start.elapsed_time(end_total)
            h2d_ms = start.elapsed_time(end_h2d)
            d2h_ms = start.elapsed_time(end_d2h)

            bidir_times.append({
                "total_ms": total_ms,
                "h2d_ms": h2d_ms,
                "d2h_ms": d2h_ms,
            })

            del gpu_result

        avg_bidir_total = np.mean([t["total_ms"] for t in bidir_times])
        avg_bidir_h2d = np.mean([t["h2d_ms"] for t in bidir_times])
        avg_bidir_d2h = np.mean([t["d2h_ms"] for t in bidir_times])

        print(f"  Bidir total: {avg_bidir_total:.3f} ms")
        print(f"  Bidir H2D: {avg_bidir_h2d:.3f} ms")
        print(f"  Bidir D2H: {avg_bidir_d2h:.3f} ms")

        sequential_time = avg_h2d + avg_d2h
        overlap_ratio = sequential_time / max(avg_bidir_total, 0.001)
        contention_h2d = avg_bidir_h2d / max(avg_h2d, 0.001)
        contention_d2h = avg_bidir_d2h / max(avg_d2h, 0.001)

        print(f"\n  Sequential (H2D + D2H): {sequential_time:.3f} ms")
        print(f"  Overlap ratio: {overlap_ratio:.2f}x")
        print(f"  H2D contention: {contention_h2d:.2f}x slowdown")
        print(f"  D2H contention: {contention_d2h:.2f}x slowdown")

        results = {
            "size_mb": size_mb,
            "h2d_only_ms": round(avg_h2d, 3),
            "h2d_only_bw_gbs": round(h2d_bw, 3),
            "d2h_only_ms": round(avg_d2h, 3),
            "d2h_only_bw_gbs": round(d2h_bw, 3),
            "bidir_total_ms": round(avg_bidir_total, 3),
            "bidir_h2d_ms": round(avg_bidir_h2d, 3),
            "bidir_d2h_ms": round(avg_bidir_d2h, 3),
            "sequential_ms": round(sequential_time, 3),
            "overlap_ratio": round(overlap_ratio, 3),
            "h2d_contention_factor": round(contention_h2d, 3),
            "d2h_contention_factor": round(contention_d2h, 3),
        }

        del cpu_tensor, cpu_h2d, gpu_d2h, cpu_d2h_dst
    except Exception as e:
        print(f"  FAILED: {e}")
        results = {"error": str(e)[:200]}

    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Experiment 5: WSL2-specific Analysis
# ============================================================

def experiment_wsl2_analysis():
    """WSL2 환경의 특이점을 분석한다."""
    print("\n" + "=" * 70)
    print("Experiment 5: WSL2 Environment Analysis")
    print("=" * 70)

    results = {}

    # System info
    sys_info = get_system_info()
    results["system_info"] = sys_info
    print(f"  Kernel: {sys_info.get('kernel', 'unknown')}")
    print(f"  PCIe: {sys_info.get('gpu_pcie', 'unknown')}")

    # GPU info
    print("\n[GPU Info]")
    props = torch.cuda.get_device_properties(0)
    results["gpu"] = {
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
    }
    print(f"  GPU: {props.name}")
    print(f"  VRAM: {props.total_memory / 1e9:.2f} GB")
    print(f"  SM count: {props.multi_processor_count}")

    # Memory pinning overhead
    print("\n[Pin Memory Overhead]")
    pin_results = []
    for size_mb in [1, 10, 100, 500]:
        n = size_mb * 1024 * 1024 // 2
        try:
            t0 = time.time()
            tensor = torch.randn(n, dtype=torch.float16)
            t1 = time.time()
            pinned = tensor.pin_memory()
            t2 = time.time()
            alloc_ms = (t1 - t0) * 1000
            pin_ms = (t2 - t1) * 1000
            pin_results.append({
                "size_mb": size_mb,
                "alloc_ms": round(alloc_ms, 2),
                "pin_ms": round(pin_ms, 2),
            })
            print(f"  {size_mb}MB: alloc={alloc_ms:.2f}ms, pin={pin_ms:.2f}ms")
            del tensor, pinned
        except Exception as e:
            print(f"  {size_mb}MB: FAILED - {e}")
            pin_results.append({"size_mb": size_mb, "error": str(e)[:200]})
        gc.collect()

    results["pin_overhead"] = pin_results

    # PCIe bandwidth analysis
    print("\n[PCIe Bandwidth Analysis]")
    theoretical_bw = 15.75  # PCIe 3.0 x16: 15.75 GB/s

    size_mb = 512
    n = size_mb * 1024 * 1024 // 2
    size_bytes = size_mb * 1024 * 1024

    try:
        cpu = torch.randn(n, dtype=torch.float16).pin_memory()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(3):
            g = cpu.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            del g
            torch.cuda.empty_cache()

        # H2D Measure
        h2d_times = []
        for _ in range(10):
            torch.cuda.empty_cache()
            start.record()
            g = cpu.to("cuda", non_blocking=True)
            end.record()
            torch.cuda.synchronize()
            h2d_times.append(start.elapsed_time(end))
            del g

        h2d_bw = size_bytes / 1e9 / (np.mean(h2d_times) / 1000)

        # D2H Measure
        g = cpu.to("cuda")
        torch.cuda.synchronize()
        d2h_times = []
        for _ in range(10):
            dst = torch.empty_like(cpu).pin_memory()
            start.record()
            dst.copy_(g, non_blocking=True)
            end.record()
            torch.cuda.synchronize()
            d2h_times.append(start.elapsed_time(end))
            del dst

        d2h_bw = size_bytes / 1e9 / (np.mean(d2h_times) / 1000)

        h2d_eff = h2d_bw / theoretical_bw * 100
        d2h_eff = d2h_bw / theoretical_bw * 100

        results["pcie_analysis"] = {
            "theoretical_bw_gbs": theoretical_bw,
            "pinned_h2d_bw_gbs": round(h2d_bw, 3),
            "pinned_d2h_bw_gbs": round(d2h_bw, 3),
            "h2d_efficiency_pct": round(h2d_eff, 1),
            "d2h_efficiency_pct": round(d2h_eff, 1),
            "h2d_wsl2_overhead_pct": round(100 - h2d_eff, 1),
            "d2h_wsl2_overhead_pct": round(100 - d2h_eff, 1),
        }
        print(f"  Theoretical PCIe 3.0 x16: {theoretical_bw} GB/s")
        print(f"  Pinned H2D: {h2d_bw:.3f} GB/s ({h2d_eff:.1f}% efficiency)")
        print(f"  Pinned D2H: {d2h_bw:.3f} GB/s ({d2h_eff:.1f}% efficiency)")
        print(f"  H2D WSL2 overhead: {100 - h2d_eff:.1f}%")
        print(f"  D2H WSL2 overhead: {100 - d2h_eff:.1f}%")

        del cpu, g
    except Exception as e:
        print(f"  PCIe test FAILED: {e}")
        results["pcie_analysis"] = {"error": str(e)[:200]}

    torch.cuda.empty_cache()
    gc.collect()

    # Pageable D2H anomaly analysis
    print("\n[Pageable D2H Anomaly Analysis]")
    print("  (방향 2에서 pageable D2H가 비정상적으로 느린 것을 관찰)")
    anomaly_results = []

    for size_mb in [64, 128, 256, 512]:
        n = size_mb * 1024 * 1024 // 2
        size_bytes = size_mb * 1024 * 1024

        try:
            # Pageable D2H
            gpu = torch.randn(n, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            pageable_times = []
            for _ in range(5):
                start.record()
                cpu_back = gpu.to("cpu")
                end.record()
                torch.cuda.synchronize()
                pageable_times.append(start.elapsed_time(end))
                del cpu_back

            # Pinned D2H
            cpu_dst = torch.empty(n, dtype=torch.float16).pin_memory()
            pinned_times = []
            for _ in range(5):
                start.record()
                cpu_dst.copy_(gpu, non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                pinned_times.append(start.elapsed_time(end))

            pageable_bw = size_bytes / 1e9 / (np.mean(pageable_times) / 1000)
            pinned_bw = size_bytes / 1e9 / (np.mean(pinned_times) / 1000)
            slowdown = np.mean(pageable_times) / np.mean(pinned_times)

            print(f"  {size_mb}MB D2H: pageable={np.mean(pageable_times):.1f}ms ({pageable_bw:.1f} GB/s), "
                  f"pinned={np.mean(pinned_times):.1f}ms ({pinned_bw:.1f} GB/s), "
                  f"slowdown={slowdown:.1f}x")

            anomaly_results.append({
                "size_mb": size_mb,
                "pageable_d2h_ms": round(np.mean(pageable_times), 3),
                "pageable_d2h_bw": round(pageable_bw, 3),
                "pinned_d2h_ms": round(np.mean(pinned_times), 3),
                "pinned_d2h_bw": round(pinned_bw, 3),
                "pageable_slowdown": round(slowdown, 2),
            })

            del gpu, cpu_dst
        except Exception as e:
            print(f"  {size_mb}MB: FAILED - {e}")
            anomaly_results.append({"size_mb": size_mb, "error": str(e)[:200]})

        torch.cuda.empty_cache()
        gc.collect()

    results["pageable_d2h_anomaly"] = anomaly_results

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(f"Part 4: WSL2 오버헤드 분석")
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

    all_results["exp1_pageable_vs_pinned"] = experiment_pageable_vs_pinned()
    all_results["exp2_large_vs_small"] = experiment_large_vs_small()
    all_results["exp3_latency"] = experiment_transfer_latency()
    all_results["exp4_bidirectional"] = experiment_bidirectional_contention()
    all_results["exp5_wsl2"] = experiment_wsl2_analysis()

    out_path = os.path.join(OUTPUT_DIR, "wsl2_overhead_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    results = main()
