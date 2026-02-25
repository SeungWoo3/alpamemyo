#!/usr/bin/env python3
"""
연구 방향 2 - Part 3: CUDA 스트림 비동기 전송 실험

목적:
  CUDA 스트림을 활용하여 모듈 N의 GPU 연산과 모듈 N+1의 CPU->GPU 전송을
  동시에 수행할 수 있는지, 그리고 그 효과를 측정한다.

실험 내용:
  1. 동기 전송 (baseline): 모듈을 순차적으로 GPU에 올리고 실행
  2. 비동기 전송: CUDA 스트림을 사용하여 다음 모듈 프리페치와 현재 모듈 실행을 중첩
  3. 더블 버퍼링: 두 개의 GPU 버퍼를 교대로 사용

제약:
  - 시스템 RAM 15GB로 전체 모델을 CPU에 올릴 수 없음
  - 따라서 실제 Alpamayo 모듈 대신 **합성(synthetic) 모듈**로 실험
  - 합성 모듈은 Alpamayo 서브모듈의 크기 특성을 모방
"""

import os
import sys
import json
import time
import csv
import gc
import threading
import math
from datetime import datetime

import torch
import torch.nn as nn


# ============================================================
# 메모리 모니터링
# ============================================================

class VRAMMonitor:
    """0.5초 간격으로 VRAM 사용량 모니터링"""

    def __init__(self, csv_path: str, interval: float = 0.5):
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
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
            else:
                allocated = reserved = 0.0
            elapsed = time.time() - self.start_time
            self.data.append({
                "time_s": round(elapsed, 2),
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
            })
            self._stop.wait(self.interval)

    def _save(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time_s", "allocated_gb", "reserved_gb"])
            writer.writeheader()
            writer.writerows(self.data)
        print(f"  [Monitor] Saved {len(self.data)} records to {self.csv_path}")


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# 합성 모듈: Alpamayo 서브모듈의 크기 특성 모방
# ============================================================

class SyntheticModule(nn.Module):
    """Alpamayo 서브모듈을 시뮬레이션하는 합성 모듈.
    지정된 파라미터 수와 연산 복잡도를 가짐."""

    def __init__(self, name: str, param_size_mb: float, hidden_dim: int = 1024,
                 compute_steps: int = 10):
        super().__init__()
        self.name = name
        self.compute_steps = compute_steps

        # 지정된 크기에 맞게 파라미터 생성 (bfloat16)
        # 1 param (bf16) = 2 bytes
        target_bytes = param_size_mb * 1e6
        n_params = int(target_bytes / 2)  # bf16

        # 여러 레이어로 분산
        n_layers = max(1, n_params // (hidden_dim * hidden_dim))
        actual_params = n_layers * hidden_dim * hidden_dim

        self.layers = nn.ModuleList()
        for _ in range(min(n_layers, 50)):  # 최대 50 레이어로 제한
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # 부족한 파라미터를 큰 버퍼로 보충
        remaining = n_params - sum(p.numel() for p in self.parameters())
        if remaining > 0:
            # 패딩 파라미터
            pad_rows = max(1, remaining // hidden_dim)
            self.padding = nn.Parameter(torch.zeros(pad_rows, hidden_dim))
        else:
            self.padding = None

        self.to(torch.bfloat16)

        actual_size = sum(p.numel() * p.element_size() for p in self.parameters()) / 1e6
        print(f"  Created {name}: target={param_size_mb:.1f}MB, actual={actual_size:.1f}MB, "
              f"params={sum(p.numel() for p in self.parameters())/1e6:.1f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for step in range(self.compute_steps):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
        return x


def create_alpamayo_proxy_modules():
    """Alpamayo의 3단계 서브모듈을 시뮬레이션하는 합성 모듈 생성.

    실제 크기:
    - Vision Encoder: 1.15 GB = 1150 MB
    - VLM (Transformer layer x1): ~386 MB
    - Expert: 4.56 GB = 4560 MB

    GPU 메모리 제약(12GB)과 시스템 RAM 제약(15GB)을 고려하여
    실제 크기의 1/10 스케일로 테스트.
    """
    scale = 0.1  # 1/10 스케일

    modules = [
        SyntheticModule("VisionEncoder", param_size_mb=1150 * scale, hidden_dim=512, compute_steps=5),
        SyntheticModule("VLM_Layer_0", param_size_mb=386 * scale, hidden_dim=512, compute_steps=8),
        SyntheticModule("VLM_Layer_1", param_size_mb=386 * scale, hidden_dim=512, compute_steps=8),
        SyntheticModule("VLM_Layer_2", param_size_mb=386 * scale, hidden_dim=512, compute_steps=8),
        SyntheticModule("Expert", param_size_mb=4560 * scale, hidden_dim=512, compute_steps=5),
    ]

    return modules


# ============================================================
# 실험 1: 동기 순차 전송 (Baseline)
# ============================================================

def test_synchronous_sequential(modules, dummy_input):
    """각 모듈을 순차적으로 GPU 로드 -> 실행 -> CPU 오프로드"""
    print("\n" + "=" * 70)
    print("Experiment 1: Synchronous Sequential Transfer (Baseline)")
    print("=" * 70)

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    results = []
    total_start = time.time()

    x = dummy_input.clone()

    for mod in modules:
        mod.cpu()  # 확실히 CPU에
    torch.cuda.synchronize()
    clear_gpu()

    for i, mod in enumerate(modules):
        phase_start = time.time()

        # CPU -> GPU 전송
        t_transfer = time.time()
        mod.to("cuda")
        torch.cuda.synchronize()
        transfer_time = time.time() - t_transfer

        vram_after_load = torch.cuda.memory_allocated(0) / 1e9

        # GPU에서 실행
        t_compute = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x_gpu = x.to("cuda")
            x_gpu = mod(x_gpu)
            x = x_gpu.cpu()
            del x_gpu
        torch.cuda.synchronize()
        compute_time = time.time() - t_compute

        vram_peak = torch.cuda.max_memory_allocated(0) / 1e9

        # GPU -> CPU 오프로드
        t_offload = time.time()
        mod.to("cpu")
        torch.cuda.synchronize()
        offload_time = time.time() - t_offload

        clear_gpu()

        phase_time = time.time() - phase_start

        result = {
            "module": mod.name,
            "transfer_time_s": round(transfer_time, 4),
            "compute_time_s": round(compute_time, 4),
            "offload_time_s": round(offload_time, 4),
            "phase_time_s": round(phase_time, 4),
            "vram_after_load_gb": round(vram_after_load, 4),
            "vram_peak_gb": round(vram_peak, 4),
        }
        results.append(result)

        print(f"  [{mod.name}] Transfer: {transfer_time:.4f}s, Compute: {compute_time:.4f}s, "
              f"Offload: {offload_time:.4f}s, Peak VRAM: {vram_peak:.4f} GB")

    total_time = time.time() - total_start
    total_transfer = sum(r["transfer_time_s"] for r in results)
    total_compute = sum(r["compute_time_s"] for r in results)
    total_offload = sum(r["offload_time_s"] for r in results)
    overall_peak = max(r["vram_peak_gb"] for r in results)

    print(f"\n  Total time: {total_time:.4f}s")
    print(f"  Total transfer: {total_transfer:.4f}s")
    print(f"  Total compute: {total_compute:.4f}s")
    print(f"  Total offload: {total_offload:.4f}s")
    print(f"  Overall peak VRAM: {overall_peak:.4f} GB")

    return {
        "method": "synchronous_sequential",
        "total_time_s": round(total_time, 4),
        "total_transfer_s": round(total_transfer, 4),
        "total_compute_s": round(total_compute, 4),
        "total_offload_s": round(total_offload, 4),
        "overall_peak_vram_gb": round(overall_peak, 4),
        "per_module": results,
    }


# ============================================================
# 실험 2: 비동기 전송 (CUDA Stream Prefetch)
# ============================================================

def test_async_prefetch(modules, dummy_input):
    """CUDA 스트림을 사용하여 다음 모듈의 프리페치와 현재 모듈의 실행을 중첩"""
    print("\n" + "=" * 70)
    print("Experiment 2: Async Prefetch with CUDA Streams")
    print("=" * 70)

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()

    # 모든 모듈을 CPU로
    for mod in modules:
        mod.cpu()
    torch.cuda.synchronize()
    clear_gpu()

    # CUDA 스트림 생성
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    results = []
    total_start = time.time()

    x = dummy_input.clone()

    # 첫 번째 모듈을 동기적으로 로드 (프리페치 대상 없음)
    t_first_transfer = time.time()
    with torch.cuda.stream(transfer_stream):
        modules[0].to("cuda")
    torch.cuda.synchronize()  # 첫 로드 완료 대기
    first_transfer_time = time.time() - t_first_transfer

    for i in range(len(modules)):
        phase_start = time.time()
        mod = modules[i]

        # 현재 모듈이 GPU에 있는지 확인
        # (첫 번째는 이미 로드됨, 이후는 프리페치로 로드됨)

        vram_after_load = torch.cuda.memory_allocated(0) / 1e9

        # 다음 모듈 프리페치 시작 (비동기)
        prefetch_started = False
        if i + 1 < len(modules):
            next_mod = modules[i + 1]
            with torch.cuda.stream(transfer_stream):
                next_mod.to("cuda", non_blocking=True)
            prefetch_started = True

        # 현재 모듈 실행 (compute stream에서)
        t_compute = time.time()
        with torch.cuda.stream(compute_stream):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x_gpu = x.to("cuda", non_blocking=True)
                compute_stream.synchronize()  # 입력 전송 완료 대기
                x_gpu = mod(x_gpu)
                x = x_gpu.cpu()
                del x_gpu

        compute_stream.synchronize()
        compute_time = time.time() - t_compute

        vram_peak = torch.cuda.max_memory_allocated(0) / 1e9

        # 현재 모듈 오프로드
        t_offload = time.time()
        mod.to("cpu")

        # 다음 모듈 프리페치 완료 대기
        if prefetch_started:
            transfer_stream.synchronize()
        offload_time = time.time() - t_offload

        # 실행 완료 후 이전 모듈의 GPU 메모리 정리
        # (다음 모듈은 이미 프리페치됨)
        gc.collect()
        torch.cuda.empty_cache()

        phase_time = time.time() - phase_start

        result = {
            "module": mod.name,
            "compute_time_s": round(compute_time, 4),
            "offload_time_s": round(offload_time, 4),
            "phase_time_s": round(phase_time, 4),
            "vram_after_load_gb": round(vram_after_load, 4),
            "vram_peak_gb": round(vram_peak, 4),
            "prefetch_overlap": prefetch_started,
        }

        if i == 0:
            result["transfer_time_s"] = round(first_transfer_time, 4)
        else:
            result["transfer_time_s"] = 0.0  # 이전 단계에서 프리페치됨

        results.append(result)

        print(f"  [{mod.name}] Compute: {compute_time:.4f}s, "
              f"Offload+Prefetch: {offload_time:.4f}s, Peak VRAM: {vram_peak:.4f} GB"
              f"{' (prefetch overlap)' if prefetch_started else ''}")

    total_time = time.time() - total_start
    total_compute = sum(r["compute_time_s"] for r in results)
    overall_peak = max(r["vram_peak_gb"] for r in results)

    print(f"\n  Total time: {total_time:.4f}s")
    print(f"  Total compute: {total_compute:.4f}s")
    print(f"  Overall peak VRAM: {overall_peak:.4f} GB")

    return {
        "method": "async_prefetch",
        "total_time_s": round(total_time, 4),
        "total_compute_s": round(total_compute, 4),
        "overall_peak_vram_gb": round(overall_peak, 4),
        "per_module": results,
        "note": "VRAM peak is higher due to overlapping current + next module"
    }


# ============================================================
# 실험 3: PCIe 대역폭 측정
# ============================================================

def test_pcie_bandwidth():
    """CPU <-> GPU 간 PCIe 대역폭을 다양한 크기로 측정"""
    print("\n" + "=" * 70)
    print("Experiment 3: PCIe Bandwidth Measurement")
    print("=" * 70)

    sizes_mb = [1, 10, 50, 100, 200, 500, 1000, 2000, 4000]
    results = []

    for size_mb in sizes_mb:
        n_elements = int(size_mb * 1e6 / 2)  # bf16 = 2 bytes
        cpu_tensor = torch.randn(n_elements, dtype=torch.bfloat16, device="cpu").pin_memory()

        # CPU -> GPU
        clear_gpu()
        torch.cuda.synchronize()
        t_start = time.time()
        gpu_tensor = cpu_tensor.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        t_h2d = time.time() - t_start

        # GPU -> CPU
        torch.cuda.synchronize()
        t_start = time.time()
        back = gpu_tensor.to("cpu", non_blocking=False)
        torch.cuda.synchronize()
        t_d2h = time.time() - t_start

        bw_h2d = size_mb / 1000 / t_h2d if t_h2d > 0 else 0  # GB/s
        bw_d2h = size_mb / 1000 / t_d2h if t_d2h > 0 else 0

        r = {
            "size_mb": size_mb,
            "h2d_time_s": round(t_h2d, 5),
            "d2h_time_s": round(t_d2h, 5),
            "h2d_bandwidth_gbps": round(bw_h2d, 2),
            "d2h_bandwidth_gbps": round(bw_d2h, 2),
        }
        results.append(r)

        print(f"  {size_mb:>5} MB | H2D: {t_h2d:.5f}s ({bw_h2d:.2f} GB/s) | "
              f"D2H: {t_d2h:.5f}s ({bw_d2h:.2f} GB/s)")

        del cpu_tensor, gpu_tensor, back
        clear_gpu()

    # 큰 전송에서의 안정 대역폭
    large_results = [r for r in results if r["size_mb"] >= 500]
    if large_results:
        avg_h2d = sum(r["h2d_bandwidth_gbps"] for r in large_results) / len(large_results)
        avg_d2h = sum(r["d2h_bandwidth_gbps"] for r in large_results) / len(large_results)
        print(f"\n  Sustained bandwidth (>= 500MB):")
        print(f"    H2D: {avg_h2d:.2f} GB/s")
        print(f"    D2H: {avg_d2h:.2f} GB/s")
    else:
        avg_h2d = avg_d2h = 0

    return {
        "method": "pcie_bandwidth",
        "measurements": results,
        "sustained_h2d_gbps": round(avg_h2d, 2),
        "sustained_d2h_gbps": round(avg_d2h, 2),
    }


# ============================================================
# 실험 4: Alpamayo 실 모듈 크기 전송 시간 추정
# ============================================================

def estimate_alpamayo_transfer_times(bandwidth_result):
    """측정된 대역폭을 바탕으로 Alpamayo 서브모듈 전송 시간을 추정"""
    print("\n" + "=" * 70)
    print("Experiment 4: Alpamayo Transfer Time Estimates")
    print("=" * 70)

    h2d_bw = bandwidth_result.get("sustained_h2d_gbps", 10.0)
    d2h_bw = bandwidth_result.get("sustained_d2h_gbps", 10.0)

    if h2d_bw == 0:
        h2d_bw = 10.0
    if d2h_bw == 0:
        d2h_bw = 10.0

    modules = {
        "Vision Encoder": {"size_gb": 1.15, "compute_time_est_s": 0.5},
        "VLM Single Layer": {"size_gb": 0.386, "compute_time_est_s": 0.05},
        "VLM Embedding+Norm": {"size_gb": 0.28, "compute_time_est_s": 0.01},
        "VLM LM Head": {"size_gb": 1.28, "compute_time_est_s": 0.02},
        "VLM Full (IMPOSSIBLE)": {"size_gb": 16.44, "compute_time_est_s": 5.0},
        "Expert Full": {"size_gb": 4.56, "compute_time_est_s": 2.0},
        "Action Projections": {"size_gb": 0.003, "compute_time_est_s": 0.001},
    }

    estimates = []
    for name, info in modules.items():
        h2d_time = info["size_gb"] / h2d_bw
        d2h_time = info["size_gb"] / d2h_bw
        total_overhead = h2d_time + d2h_time

        est = {
            "module": name,
            "size_gb": info["size_gb"],
            "h2d_time_s": round(h2d_time, 4),
            "d2h_time_s": round(d2h_time, 4),
            "total_transfer_overhead_s": round(total_overhead, 4),
            "compute_time_est_s": info["compute_time_est_s"],
            "can_hide_transfer": h2d_time < info["compute_time_est_s"],
        }
        estimates.append(est)

        print(f"  {name:25s} | {info['size_gb']:6.3f} GB | H2D: {h2d_time:.4f}s | "
              f"D2H: {d2h_time:.4f}s | Hideable: {'YES' if est['can_hide_transfer'] else 'NO'}")

    # VLM 레이어별 오프로딩 시나리오
    vlm_layer = next(e for e in estimates if e["module"] == "VLM Single Layer")
    n_vlm_layers = 36

    # 동기 시나리오
    sync_vlm_time = n_vlm_layers * (vlm_layer["h2d_time_s"] + vlm_layer["compute_time_est_s"] + vlm_layer["d2h_time_s"])

    # 비동기 시나리오 (파이프라인으로 전송 은닉)
    pipeline_overhead = vlm_layer["h2d_time_s"]  # 첫 레이어 cold start
    for i in range(n_vlm_layers):
        step_time = max(vlm_layer["h2d_time_s"], vlm_layer["compute_time_est_s"])
        pipeline_overhead += step_time
    pipeline_overhead += vlm_layer["d2h_time_s"]  # 마지막 레이어 cleanup

    # Expert 10-step diffusion
    expert = next(e for e in estimates if e["module"] == "Expert Full")
    expert_sync = expert["h2d_time_s"] + 10 * expert["compute_time_est_s"] + expert["d2h_time_s"]

    print(f"\n  --- VLM Layer-wise Offloading ({n_vlm_layers} layers) ---")
    print(f"  Synchronous: {sync_vlm_time:.2f}s")
    print(f"  Pipelined (async prefetch): {pipeline_overhead:.2f}s")
    print(f"  Speedup from pipelining: {sync_vlm_time / pipeline_overhead:.2f}x")

    print(f"\n  --- Expert with 10-step Diffusion ---")
    print(f"  Expert (loaded once, 10 steps): {expert_sync:.2f}s")

    # 전체 추론 시간 추정
    vision = next(e for e in estimates if e["module"] == "Vision Encoder")
    lm_head = next(e for e in estimates if e["module"] == "VLM LM Head")
    embed = next(e for e in estimates if e["module"] == "VLM Embedding+Norm")

    total_sync = (vision["total_transfer_overhead_s"] + vision["compute_time_est_s"] +
                  embed["total_transfer_overhead_s"] + embed["compute_time_est_s"] +
                  sync_vlm_time +
                  lm_head["total_transfer_overhead_s"] + lm_head["compute_time_est_s"] +
                  expert_sync)

    total_async = (vision["total_transfer_overhead_s"] + vision["compute_time_est_s"] +
                   embed["total_transfer_overhead_s"] + embed["compute_time_est_s"] +
                   pipeline_overhead +
                   lm_head["total_transfer_overhead_s"] + lm_head["compute_time_est_s"] +
                   expert_sync)

    print(f"\n  --- Total Inference Time Estimates ---")
    print(f"  Baseline (GPU-only, no offload): ~273.79s (from prior experiments)")
    print(f"  Sequential offload (sync): ~{total_sync:.2f}s overhead + compute")
    print(f"  Sequential offload (async pipeline): ~{total_async:.2f}s overhead + compute")

    return {
        "per_module_estimates": estimates,
        "vlm_layerwise": {
            "n_layers": n_vlm_layers,
            "sync_total_s": round(sync_vlm_time, 2),
            "pipeline_total_s": round(pipeline_overhead, 2),
            "speedup": round(sync_vlm_time / pipeline_overhead, 2) if pipeline_overhead > 0 else 0,
        },
        "expert_10step": {
            "total_s": round(expert_sync, 2),
        },
        "total_inference_est": {
            "baseline_s": 273.79,
            "sync_offload_overhead_s": round(total_sync, 2),
            "async_offload_overhead_s": round(total_async, 2),
        },
        "bandwidth_used": {
            "h2d_gbps": h2d_bw,
            "d2h_gbps": d2h_bw,
        },
    }


# ============================================================
# 메인 실험
# ============================================================

def main():
    print("=" * 70)
    print("CUDA Stream Async Transfer Experiment")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 70)

    output_dir = "/home/seungwoo/workspace/research/02-memory-pipelining"
    csv_path = os.path.join(output_dir, "async_transfer_vram_timeline.csv")
    results_path = os.path.join(output_dir, "async_transfer_results.json")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # VRAM 모니터 시작
    monitor = VRAMMonitor(csv_path)
    monitor.start()

    results = {
        "experiment": "cuda_stream_async_transfer",
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 1),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "experiments": [],
    }

    # ----------------------------------------------------------
    # 합성 모듈 생성
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Creating Synthetic Proxy Modules (1/10 scale)")
    print("=" * 70)

    modules = create_alpamayo_proxy_modules()

    # 더미 입력
    dummy_input = torch.randn(1, 512, dtype=torch.bfloat16, device="cpu")

    # ----------------------------------------------------------
    # 실험 1: 동기 순차 전송
    # ----------------------------------------------------------
    try:
        sync_result = test_synchronous_sequential(modules, dummy_input)
        results["experiments"].append(sync_result)
    except Exception as e:
        print(f"  Experiment 1 FAILED: {e}")
        results["experiments"].append({"method": "synchronous_sequential", "error": str(e)})

    gc.collect()
    clear_gpu()

    # 모듈을 다시 CPU로
    for mod in modules:
        mod.cpu()
    gc.collect()

    # ----------------------------------------------------------
    # 실험 2: 비동기 프리페치
    # ----------------------------------------------------------
    try:
        async_result = test_async_prefetch(modules, dummy_input)
        results["experiments"].append(async_result)
    except Exception as e:
        print(f"  Experiment 2 FAILED: {e}")
        results["experiments"].append({"method": "async_prefetch", "error": str(e)})

    gc.collect()
    clear_gpu()

    # 합성 모듈 삭제하여 RAM 확보
    del modules
    gc.collect()

    # ----------------------------------------------------------
    # 실험 3: PCIe 대역폭 측정
    # ----------------------------------------------------------
    try:
        bw_result = test_pcie_bandwidth()
        results["experiments"].append(bw_result)
    except Exception as e:
        print(f"  Experiment 3 FAILED: {e}")
        bw_result = {"sustained_h2d_gbps": 10.0, "sustained_d2h_gbps": 10.0}
        results["experiments"].append({"method": "pcie_bandwidth", "error": str(e)})

    # ----------------------------------------------------------
    # 실험 4: Alpamayo 전송 시간 추정
    # ----------------------------------------------------------
    try:
        transfer_est = estimate_alpamayo_transfer_times(bw_result)
        results["experiments"].append({"method": "transfer_estimates", **transfer_est})
    except Exception as e:
        print(f"  Experiment 4 FAILED: {e}")
        results["experiments"].append({"method": "transfer_estimates", "error": str(e)})

    # ----------------------------------------------------------
    # 비교 분석
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)

    sync_exp = next((e for e in results["experiments"] if e.get("method") == "synchronous_sequential"), None)
    async_exp = next((e for e in results["experiments"] if e.get("method") == "async_prefetch"), None)

    if sync_exp and async_exp and "total_time_s" in sync_exp and "total_time_s" in async_exp:
        sync_time = sync_exp["total_time_s"]
        async_time = async_exp["total_time_s"]
        speedup = sync_time / async_time if async_time > 0 else 0

        sync_peak = sync_exp.get("overall_peak_vram_gb", 0)
        async_peak = async_exp.get("overall_peak_vram_gb", 0)

        print(f"  Synchronous:  {sync_time:.4f}s, Peak VRAM: {sync_peak:.4f} GB")
        print(f"  Async:        {async_time:.4f}s, Peak VRAM: {async_peak:.4f} GB")
        print(f"  Speedup:      {speedup:.2f}x")
        print(f"  VRAM overhead: {async_peak - sync_peak:.4f} GB (async uses more due to overlap)")

        results["comparison"] = {
            "sync_time_s": sync_time,
            "async_time_s": async_time,
            "speedup": round(speedup, 3),
            "sync_peak_vram_gb": sync_peak,
            "async_peak_vram_gb": async_peak,
            "vram_overhead_gb": round(async_peak - sync_peak, 4),
        }
    else:
        print("  Could not compare (one or both experiments failed)")

    # 모니터 종료
    monitor.stop()

    # 결과 저장
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"VRAM timeline saved to {csv_path}")

    return results


if __name__ == "__main__":
    main()
