#!/usr/bin/env python3
"""Part 3: 명시적 텐서 핀닝 + 비동기 전송 실험

Unified Memory 대신 명시적으로 관리하는 방법:
1. 모델 가중치를 pinned memory에 배치
2. CUDA 스트림으로 비동기 전송
3. non_blocking=True 전송 vs pageable 성능 비교
4. 전송 크기별 대역폭 측정 (pinned vs non-pinned)

NOTE: WSL2에서 cudaMemPrefetchAsync(cpu) 호출이 CUDA context를
오염시키므로, Unified Memory 관련 ctypes 호출은 완전히 제외한다.
"""

import os
import sys
import json
import time
import gc
import csv
import threading
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

OUTPUT_DIR = "/home/seungwoo/workspace/research/03-swap-optimization"


# ============================================================
# VRAM Monitor
# ============================================================

class VRAMMonitor:
    """50ms 간격 VRAM 모니터링"""

    def __init__(self, csv_path: str, interval: float = 0.05):
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
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                elapsed = time.time() - self.start_time
                self.data.append({
                    "time_s": round(elapsed, 3),
                    "allocated_gb": round(allocated, 4),
                    "reserved_gb": round(reserved, 4),
                })
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _save(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time_s", "allocated_gb", "reserved_gb"])
            writer.writeheader()
            writer.writerows(self.data)


# ============================================================
# Experiment 1: Bandwidth Comparison (Pinned vs Pageable)
# ============================================================

def experiment_bandwidth_comparison():
    """전송 크기별 대역폭을 pinned/pageable memory에서 비교한다."""
    print("=" * 70)
    print("Experiment 1: Bandwidth Comparison (Pinned vs Pageable)")
    print("=" * 70)

    results = []
    sizes_mb = [64, 128, 256, 512, 1024]
    n_warmup = 2
    n_trials = 5

    for size_mb in sizes_mb:
        n_elements = size_mb * 1024 * 1024 // 2  # FP16
        size_bytes = size_mb * 1024 * 1024
        print(f"\n--- {size_mb} MB ---")

        entry = {"size_mb": size_mb}

        # ==== Method 1: Pinned Memory (H2D) ====
        try:
            cpu_pinned = torch.randn(n_elements, dtype=torch.float16).pin_memory()

            h2d_times = []
            for i in range(n_warmup + n_trials):
                torch.cuda.empty_cache()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu_tensor = cpu_pinned.to("cuda", non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    h2d_times.append(start.elapsed_time(end))
                del gpu_tensor

            avg_h2d = np.mean(h2d_times)
            h2d_bw = size_bytes / 1e9 / (avg_h2d / 1000)
            entry["pinned_h2d_ms"] = round(avg_h2d, 3)
            entry["pinned_h2d_bw_gbs"] = round(h2d_bw, 3)
            print(f"  Pinned H2D: {avg_h2d:.3f} ms ({h2d_bw:.2f} GB/s)")

            # Pinned Memory (D2H)
            gpu_tensor = cpu_pinned.to("cuda")
            torch.cuda.synchronize()

            d2h_times = []
            for i in range(n_warmup + n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                cpu_dst = torch.empty_like(cpu_pinned).pin_memory()
                start.record()
                cpu_dst.copy_(gpu_tensor, non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    d2h_times.append(start.elapsed_time(end))
                del cpu_dst

            avg_d2h = np.mean(d2h_times)
            d2h_bw = size_bytes / 1e9 / (avg_d2h / 1000)
            entry["pinned_d2h_ms"] = round(avg_d2h, 3)
            entry["pinned_d2h_bw_gbs"] = round(d2h_bw, 3)
            print(f"  Pinned D2H: {avg_d2h:.3f} ms ({d2h_bw:.2f} GB/s)")

            del gpu_tensor, cpu_pinned
        except Exception as e:
            print(f"  Pinned: FAILED - {e}")
            entry["pinned_error"] = str(e)[:200]

        torch.cuda.empty_cache()
        gc.collect()

        # ==== Method 2: Pageable Memory (H2D) ====
        try:
            cpu_pageable = torch.randn(n_elements, dtype=torch.float16)

            h2d_times = []
            for i in range(n_warmup + n_trials):
                torch.cuda.empty_cache()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                gpu_tensor = cpu_pageable.to("cuda", non_blocking=False)
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    h2d_times.append(start.elapsed_time(end))
                del gpu_tensor

            avg_h2d = np.mean(h2d_times)
            h2d_bw = size_bytes / 1e9 / (avg_h2d / 1000)
            entry["pageable_h2d_ms"] = round(avg_h2d, 3)
            entry["pageable_h2d_bw_gbs"] = round(h2d_bw, 3)
            print(f"  Pageable H2D: {avg_h2d:.3f} ms ({h2d_bw:.2f} GB/s)")

            # Pageable Memory (D2H)
            gpu_tensor = cpu_pageable.to("cuda")
            torch.cuda.synchronize()

            d2h_times = []
            for i in range(n_warmup + n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                cpu_back = gpu_tensor.to("cpu")
                end.record()
                torch.cuda.synchronize()
                if i >= n_warmup:
                    d2h_times.append(start.elapsed_time(end))
                del cpu_back

            avg_d2h = np.mean(d2h_times)
            d2h_bw = size_bytes / 1e9 / (avg_d2h / 1000)
            entry["pageable_d2h_ms"] = round(avg_d2h, 3)
            entry["pageable_d2h_bw_gbs"] = round(d2h_bw, 3)
            print(f"  Pageable D2H: {avg_d2h:.3f} ms ({d2h_bw:.2f} GB/s)")

            del gpu_tensor, cpu_pageable
        except Exception as e:
            print(f"  Pageable: FAILED - {e}")
            entry["pageable_error"] = str(e)[:200]

        torch.cuda.empty_cache()
        gc.collect()

        # Calculate speedups
        if "pinned_h2d_bw_gbs" in entry and "pageable_h2d_bw_gbs" in entry:
            entry["h2d_pinned_speedup"] = round(entry["pinned_h2d_bw_gbs"] / max(entry["pageable_h2d_bw_gbs"], 0.001), 2)
        if "pinned_d2h_bw_gbs" in entry and "pageable_d2h_bw_gbs" in entry:
            entry["d2h_pinned_speedup"] = round(entry["pinned_d2h_bw_gbs"] / max(entry["pageable_d2h_bw_gbs"], 0.001), 2)

        results.append(entry)

    return results


# ============================================================
# Experiment 2: Async Stream Transfer vs Blocking Transfer
# ============================================================

def experiment_async_vs_blocking():
    """비동기 스트림 전송 vs 블로킹 전송을 비교한다."""
    print("\n" + "=" * 70)
    print("Experiment 2: Async Stream Transfer vs Blocking Transfer")
    print("=" * 70)

    results = []

    # Create synthetic "model layers" on CPU (pinned)
    # NOTE: Using 200MB layers instead of 386MB to fit in 15GB system RAM
    # (5 x 386MB pinned = 1.93GB pinned, plus overhead, triggers OOM killer)
    layer_sizes_mb = [200, 200, 200, 200, 200]  # 5 scaled layers
    total_layers = len(layer_sizes_mb)

    print(f"\nSimulating {total_layers} VLM layers of ~386 MB each")

    # Create pinned CPU tensors
    cpu_layers = []
    for size_mb in layer_sizes_mb:
        n_elements = size_mb * 1024 * 1024 // 2
        layer = torch.randn(n_elements, dtype=torch.float16).pin_memory()
        cpu_layers.append(layer)

    # Test 1: Blocking sequential transfer
    print("\n[Test 1] Blocking sequential transfer...")
    torch.cuda.empty_cache()
    gc.collect()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for layer in cpu_layers:
        gpu_layer = layer.to("cuda", non_blocking=False)
        # Simulate some computation
        _ = gpu_layer.sum()
        torch.cuda.synchronize()
        del gpu_layer
        torch.cuda.empty_cache()
    end.record()
    torch.cuda.synchronize()
    blocking_ms = start.elapsed_time(end)
    print(f"  Blocking: {blocking_ms:.3f} ms")

    # Test 2: Non-blocking with explicit synchronization
    print("\n[Test 2] Non-blocking transfer...")
    torch.cuda.empty_cache()
    gc.collect()

    start.record()
    for layer in cpu_layers:
        gpu_layer = layer.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        _ = gpu_layer.sum()
        torch.cuda.synchronize()
        del gpu_layer
        torch.cuda.empty_cache()
    end.record()
    torch.cuda.synchronize()
    nonblocking_ms = start.elapsed_time(end)
    print(f"  Non-blocking: {nonblocking_ms:.3f} ms")

    # Test 3: Double-buffered async pipeline
    print("\n[Test 3] Double-buffered async pipeline...")
    torch.cuda.empty_cache()
    gc.collect()

    transfer_stream = torch.cuda.Stream()

    start.record()

    # Prefetch first layer
    with torch.cuda.stream(transfer_stream):
        gpu_buf = cpu_layers[0].to("cuda", non_blocking=True)

    for i in range(total_layers):
        # Wait for current layer transfer to complete
        torch.cuda.current_stream().wait_stream(transfer_stream)
        current_gpu = gpu_buf

        # Start prefetching next layer (if any)
        if i + 1 < total_layers:
            with torch.cuda.stream(transfer_stream):
                gpu_buf = cpu_layers[i + 1].to("cuda", non_blocking=True)

        # Compute on current layer
        _ = current_gpu.sum()
        torch.cuda.synchronize()
        del current_gpu
        torch.cuda.empty_cache()

    end.record()
    torch.cuda.synchronize()
    pipeline_ms = start.elapsed_time(end)
    print(f"  Pipeline: {pipeline_ms:.3f} ms")

    results.append({
        "n_layers": total_layers,
        "layer_size_mb": layer_sizes_mb[0],
        "blocking_ms": round(blocking_ms, 3),
        "nonblocking_ms": round(nonblocking_ms, 3),
        "pipeline_ms": round(pipeline_ms, 3),
        "pipeline_speedup_vs_blocking": round(blocking_ms / max(pipeline_ms, 0.001), 3),
    })

    del cpu_layers
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Experiment 3: Simulated Layer-wise Offloading
# ============================================================

def experiment_layerwise_offload():
    """레이어별 오프로딩을 시뮬레이션하여 실제 추론 시나리오를 테스트한다."""
    print("\n" + "=" * 70)
    print("Experiment 3: Simulated Layer-wise Offloading")
    print("=" * 70)

    results = {}

    n_layers = 6  # Reduced for memory safety
    hidden_dim = 2048  # Reduced from 4096 to fit in 15GB RAM
    intermediate_dim = 7168  # Reduced proportionally

    # Create simple linear layers on CPU with pinned memory
    class SimpleLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim // 8, bias=False)  # GQA
            self.v_proj = nn.Linear(hidden_dim, hidden_dim // 8, bias=False)
            self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
            self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    # Create layers
    print(f"Creating {n_layers} layers on CPU...")
    cpu_layers = []
    for i in range(n_layers):
        layer = SimpleLayer().half()
        # Pin memory for each parameter
        for param in layer.parameters():
            param.data = param.data.pin_memory()
        cpu_layers.append(layer)

    layer_size_bytes = sum(p.numel() * p.element_size() for p in cpu_layers[0].parameters())
    layer_size_mb = layer_size_bytes / (1024 * 1024)
    print(f"Layer size: {layer_size_mb:.1f} MB ({n_layers} layers)")

    # Create dummy input
    batch_size = 1
    seq_len = 128
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

    # ---- Strategy 1: Sync offloading ----
    print("\n[Strategy 1] Sync layer-wise offloading...")
    torch.cuda.empty_cache()
    gc.collect()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    layer_times_sync = []

    start.record()
    h = x.clone()
    for i, layer in enumerate(cpu_layers):
        t0 = time.time()
        # Move layer to GPU
        layer_gpu = SimpleLayer().half().cuda()
        for (n1, p1), (n2, p2) in zip(layer_gpu.named_parameters(), layer.named_parameters()):
            p1.data.copy_(p2.data, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()

        # Forward pass (simplified)
        with torch.no_grad():
            q = layer_gpu.q_proj(h)
            k = layer_gpu.k_proj(h)
            v = layer_gpu.v_proj(h)
            h = layer_gpu.o_proj(q)
            gate = layer_gpu.gate_proj(h)
            up = layer_gpu.up_proj(h)
            h = layer_gpu.down_proj(gate * up)
        torch.cuda.synchronize()
        t2 = time.time()

        # Free GPU layer
        del layer_gpu
        torch.cuda.empty_cache()
        t3 = time.time()

        layer_times_sync.append({
            "layer": i,
            "h2d_ms": round((t1 - t0) * 1000, 2),
            "compute_ms": round((t2 - t1) * 1000, 2),
            "cleanup_ms": round((t3 - t2) * 1000, 2),
        })

    end.record()
    torch.cuda.synchronize()
    sync_total_ms = start.elapsed_time(end)
    print(f"  Total: {sync_total_ms:.3f} ms")
    print(f"  Avg H2D: {np.mean([t['h2d_ms'] for t in layer_times_sync]):.2f} ms")
    print(f"  Avg Compute: {np.mean([t['compute_ms'] for t in layer_times_sync]):.2f} ms")

    results["sync"] = {
        "total_ms": round(sync_total_ms, 3),
        "layer_times": layer_times_sync,
    }

    # ---- Strategy 2: Async pipeline offloading ----
    print("\n[Strategy 2] Async pipeline offloading...")
    torch.cuda.empty_cache()
    gc.collect()

    transfer_stream = torch.cuda.Stream()
    layer_times_async = []

    start.record()
    h = x.clone()

    # Pre-allocate GPU buffer and copy first layer
    next_layer_gpu = SimpleLayer().half().cuda()
    with torch.cuda.stream(transfer_stream):
        for (n1, p1), (n2, p2) in zip(next_layer_gpu.named_parameters(), cpu_layers[0].named_parameters()):
            p1.data.copy_(p2.data, non_blocking=True)

    for i in range(n_layers):
        t0 = time.time()
        # Wait for transfer to complete
        torch.cuda.current_stream().wait_stream(transfer_stream)
        current_layer_gpu = next_layer_gpu
        t1 = time.time()

        # Start transferring next layer
        if i + 1 < n_layers:
            next_layer_gpu = SimpleLayer().half().cuda()
            with torch.cuda.stream(transfer_stream):
                for (n1, p1), (n2, p2) in zip(next_layer_gpu.named_parameters(), cpu_layers[i + 1].named_parameters()):
                    p1.data.copy_(p2.data, non_blocking=True)

        # Forward pass on current layer
        with torch.no_grad():
            q = current_layer_gpu.q_proj(h)
            k = current_layer_gpu.k_proj(h)
            v = current_layer_gpu.v_proj(h)
            h = current_layer_gpu.o_proj(q)
            gate = current_layer_gpu.gate_proj(h)
            up = current_layer_gpu.up_proj(h)
            h = current_layer_gpu.down_proj(gate * up)
        torch.cuda.synchronize()
        t2 = time.time()

        # Free current layer
        del current_layer_gpu
        torch.cuda.empty_cache()
        t3 = time.time()

        layer_times_async.append({
            "layer": i,
            "wait_ms": round((t1 - t0) * 1000, 2),
            "compute_ms": round((t2 - t1) * 1000, 2),
            "cleanup_ms": round((t3 - t2) * 1000, 2),
        })

    end.record()
    torch.cuda.synchronize()
    async_total_ms = start.elapsed_time(end)
    print(f"  Total: {async_total_ms:.3f} ms")
    print(f"  Avg Wait: {np.mean([t['wait_ms'] for t in layer_times_async]):.2f} ms")
    print(f"  Avg Compute: {np.mean([t['compute_ms'] for t in layer_times_async]):.2f} ms")

    results["async_pipeline"] = {
        "total_ms": round(async_total_ms, 3),
        "layer_times": layer_times_async,
    }

    speedup = sync_total_ms / max(async_total_ms, 0.001)
    print(f"\n  Pipeline speedup: {speedup:.2f}x")

    results["comparison"] = {
        "n_layers": n_layers,
        "layer_size_mb": round(layer_size_mb, 1),
        "sync_total_ms": round(sync_total_ms, 3),
        "async_total_ms": round(async_total_ms, 3),
        "speedup": round(speedup, 3),
    }

    # Cleanup
    del cpu_layers, x
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# Experiment 4: Transfer Size Scaling
# ============================================================

def experiment_transfer_size_scaling():
    """전송 크기 vs 대역폭 효율의 스케일링을 분석한다."""
    print("\n" + "=" * 70)
    print("Experiment 4: Transfer Size Scaling")
    print("=" * 70)

    results = []
    n_trials = 5

    for size_mb in [1, 4, 16, 64, 128, 256, 512, 1024]:
        n_elements = size_mb * 1024 * 1024 // 2  # FP16
        size_bytes = size_mb * 1024 * 1024
        print(f"\n--- {size_mb} MB ---")

        entry = {"size_mb": size_mb}

        try:
            # Pinned H2D
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
            std_h2d = np.std(h2d_times)
            h2d_bw = size_bytes / 1e9 / (avg_h2d / 1000)
            entry["h2d_ms"] = round(avg_h2d, 3)
            entry["h2d_std_ms"] = round(std_h2d, 3)
            entry["h2d_bw_gbs"] = round(h2d_bw, 3)

            # Pinned D2H
            gpu_tensor = cpu_tensor.to("cuda")
            torch.cuda.synchronize()

            d2h_times = []
            for i in range(n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                cpu_dst = torch.empty_like(cpu_tensor).pin_memory()
                start.record()
                cpu_dst.copy_(gpu_tensor, non_blocking=True)
                end.record()
                torch.cuda.synchronize()
                d2h_times.append(start.elapsed_time(end))
                del cpu_dst

            avg_d2h = np.mean(d2h_times)
            std_d2h = np.std(d2h_times)
            d2h_bw = size_bytes / 1e9 / (avg_d2h / 1000)
            entry["d2h_ms"] = round(avg_d2h, 3)
            entry["d2h_std_ms"] = round(std_d2h, 3)
            entry["d2h_bw_gbs"] = round(d2h_bw, 3)

            entry["asymmetry_ratio"] = round(h2d_bw / max(d2h_bw, 0.001), 2)

            print(f"  Pinned H2D: {avg_h2d:.3f}ms ({h2d_bw:.2f} GB/s), "
                  f"D2H: {avg_d2h:.3f}ms ({d2h_bw:.2f} GB/s), "
                  f"Asymmetry: {entry['asymmetry_ratio']:.2f}x")

            del cpu_tensor, gpu_tensor
        except Exception as e:
            print(f"  FAILED: {e}")
            entry["error"] = str(e)[:200]

        torch.cuda.empty_cache()
        gc.collect()
        results.append(entry)

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print(f"Part 3: 명시적 텐서 핀닝 + 비동기 전송 실험")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70)

    monitor = VRAMMonitor(os.path.join(OUTPUT_DIR, "pinned_transfer_vram.csv"))
    monitor.start()

    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
        }
    }

    all_results["exp1_bandwidth"] = experiment_bandwidth_comparison()
    all_results["exp2_async_vs_blocking"] = experiment_async_vs_blocking()
    all_results["exp3_layerwise"] = experiment_layerwise_offload()
    all_results["exp4_size_scaling"] = experiment_transfer_size_scaling()

    monitor.stop()

    out_path = os.path.join(OUTPUT_DIR, "pinned_transfer_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    results = main()
