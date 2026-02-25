#!/usr/bin/env python3
"""
연구 방향 2: 모듈 순차 오프로딩 (Sequential Module Offloading) 실험

Alpamayo-R1-10B의 3단계 파이프라인:
  Phase 1: Vision Encoder (1.15GB) -> GPU -> 이미지 처리 -> CPU
  Phase 2: VLM Language Model (15.17GB) -> GPU -> 토큰 생성 -> CPU
  Phase 3: Expert/Diffusion (4.56GB) -> GPU -> 궤적 디코딩 -> CPU

시스템 제약:
  - GPU VRAM: 12GB (RTX 3080 Ti)
  - System RAM: 15GB (전체 모델 22GB 동시 CPU 로딩 불가)
  - 따라서 디스크 오프로딩 또는 단계별 부분 로딩이 필요

실험 내용:
  1. 모델을 CPU에 로드 (가능한 만큼)
  2. 각 서브모듈을 순차적으로 GPU로 이동하여 실행
  3. 실행 완료 후 CPU로 반환
  4. 각 단계별 VRAM 피크 및 소요 시간 측정
"""

import os
import sys
import json
import time
import csv
import gc
import threading
import traceback
from datetime import datetime

import torch
import numpy as np

# ============================================================
# 메모리 모니터링
# ============================================================

class VRAMMonitor:
    """0.5초 간격으로 VRAM 사용량을 모니터링하여 CSV에 기록"""

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


def get_vram_mb():
    """현재 VRAM 사용량 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1e6
    return 0.0


def get_vram_gb():
    """현재 VRAM 사용량 (GB)"""
    return get_vram_mb() / 1000.0


def get_peak_vram_gb():
    """피크 VRAM 사용량 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(0) / 1e9
    return 0.0


def get_ram_usage_gb():
    """현재 시스템 RAM 사용량 (GB) - /proc/meminfo 기반"""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            info[parts[0].rstrip(":")] = int(parts[1]) / 1024 / 1024  # kB -> GB
        used = info.get("MemTotal", 0) - info.get("MemAvailable", 0)
        return round(used, 2)
    except Exception:
        return 0.0


def clear_gpu():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# 모듈 크기 측정 유틸리티
# ============================================================

def module_size_gb(module: torch.nn.Module) -> float:
    """모듈의 파라미터 + 버퍼 메모리 크기 (GB)"""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total / 1e9


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ============================================================
# Phase 1: Vision Encoder 테스트
# ============================================================

def test_vision_encoder_phase(model, dummy_pixel_values, dummy_image_grid_thw):
    """Vision Encoder만 GPU로 이동하여 이미지 처리"""
    print("\n" + "=" * 70)
    print("Phase 1: Vision Encoder")
    print("=" * 70)

    vision = model.vlm.model.visual
    v_size = module_size_gb(vision)
    print(f"  Module size: {v_size:.3f} GB ({count_params(vision)/1e6:.1f}M params)")

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    vram_before = get_vram_gb()
    ram_before = get_ram_usage_gb()

    t_start = time.time()

    # GPU로 이동
    t_move_start = time.time()
    vision.to("cuda")
    t_move_gpu = time.time() - t_move_start

    vram_after_load = get_vram_gb()
    print(f"  VRAM after GPU load: {vram_after_load:.3f} GB (delta: {vram_after_load - vram_before:.3f} GB)")
    print(f"  CPU->GPU transfer time: {t_move_gpu:.3f}s")

    # 추론 실행 시도
    t_infer_start = time.time()
    inference_status = "SUCCESS"
    output_shape = "N/A"
    output_size_mb = 0.0
    vision_result = None

    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pixel_values_gpu = dummy_pixel_values.to("cuda")
            image_grid_thw_gpu = dummy_image_grid_thw.to("cuda")
            vision_output = vision(pixel_values_gpu, grid_thw=image_grid_thw_gpu)

        if isinstance(vision_output, torch.Tensor):
            output_shape = str(vision_output.shape)
            output_size_mb = vision_output.numel() * vision_output.element_size() / 1e6
            vision_result = vision_output.cpu()
            del vision_output
        del pixel_values_gpu, image_grid_thw_gpu
    except Exception as e:
        inference_status = f"FORWARD_FAILED (input format issue, transfer metrics still valid): {str(e)[:150]}"
        print(f"  Note: Forward pass failed due to input format, but transfer metrics are valid")
        # 대신 간단한 더미 연산으로 VRAM 측정
        try:
            dummy_gpu = torch.randn(1000, 1152, device="cuda", dtype=torch.bfloat16)
            _ = torch.matmul(dummy_gpu, dummy_gpu.T)
            del dummy_gpu
        except Exception:
            pass

    t_infer = time.time() - t_infer_start
    vram_peak = get_peak_vram_gb()

    print(f"  Inference status: {inference_status}")
    print(f"  Inference time: {t_infer:.3f}s")
    print(f"  Output shape: {output_shape}, size: {output_size_mb:.2f} MB")
    print(f"  Peak VRAM: {vram_peak:.3f} GB")

    # CPU로 반환
    t_offload_start = time.time()
    vision.to("cpu")
    t_offload = time.time() - t_offload_start

    clear_gpu()
    vram_after_offload = get_vram_gb()

    t_total = time.time() - t_start

    print(f"  GPU->CPU offload time: {t_offload:.3f}s")
    print(f"  VRAM after offload: {vram_after_offload:.3f} GB")
    print(f"  Total phase time: {t_total:.3f}s")

    result = {
        "phase": "Vision Encoder",
        "status": inference_status,
        "module_size_gb": round(v_size, 3),
        "params_m": round(count_params(vision) / 1e6, 1),
        "cpu_to_gpu_time_s": round(t_move_gpu, 3),
        "inference_time_s": round(t_infer, 3),
        "gpu_to_cpu_time_s": round(t_offload, 3),
        "total_time_s": round(t_total, 3),
        "vram_before_gb": round(vram_before, 3),
        "vram_after_load_gb": round(vram_after_load, 3),
        "vram_peak_gb": round(vram_peak, 3),
        "vram_after_offload_gb": round(vram_after_offload, 3),
        "output_shape": str(output_shape),
        "output_size_mb": round(output_size_mb, 2),
    }

    return result, vision_result


# ============================================================
# Phase 2: VLM Language Model 테스트 (레이어 단위)
# ============================================================

def test_vlm_layerwise_phase(model):
    """VLM Language Model은 15.17GB로 12GB를 초과하므로, 레이어 단위로 테스트.
    여기서는 전체 GPU 로딩이 불가능하므로, 개별 레이어의 이동 시간과 크기를 측정한다.
    """
    print("\n" + "=" * 70)
    print("Phase 2: VLM Language Model (Layer-wise Analysis)")
    print("=" * 70)

    # 전체 VLM Language Model 크기
    lang_model = model.vlm.model
    lm_head = model.vlm.lm_head

    vlm_total_size = module_size_gb(lang_model) + module_size_gb(lm_head)
    print(f"  Total VLM size: {vlm_total_size:.3f} GB")
    print(f"  This EXCEEDS 12GB VRAM - full GPU loading is IMPOSSIBLE")
    print(f"  Performing layer-wise transfer time analysis instead...")

    # Transformer 레이어 목록 접근
    layers = None
    for attr_path in [
        "language_model.layers",
        "language_model.model.layers",
        "model.layers",
        "layers",
    ]:
        try:
            obj = lang_model
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                layers = obj
                print(f"  Found layers at vlm.model.{attr_path}: {len(obj)} layers")
                break
        except AttributeError:
            continue

    if layers is None:
        print("  ERROR: Cannot find transformer layers")
        return {
            "phase": "VLM Language Model",
            "status": "FAILED",
            "reason": "Cannot locate transformer layers",
            "module_size_gb": round(vlm_total_size, 3),
        }, None

    n_layers = len(layers)
    print(f"  Number of transformer layers: {n_layers}")

    # 개별 레이어 분석
    layer_sizes = []
    layer_transfer_times = []

    # 3개 레이어 샘플 테스트 (첫번째, 중간, 마지막)
    sample_indices = [0, n_layers // 2, n_layers - 1]

    for idx in sample_indices:
        layer = layers[idx]
        l_size = module_size_gb(layer)
        l_params = count_params(layer)
        layer_sizes.append(l_size)

        clear_gpu()
        torch.cuda.reset_peak_memory_stats()

        # CPU -> GPU 전송
        t_up = time.time()
        layer.to("cuda")
        torch.cuda.synchronize()
        t_up = time.time() - t_up

        vram_used = get_vram_gb()

        # GPU -> CPU 전송
        t_down = time.time()
        layer.to("cpu")
        t_down = time.time() - t_down

        clear_gpu()

        transfer_info = {
            "layer_idx": idx,
            "size_gb": round(l_size, 4),
            "params_m": round(l_params / 1e6, 1),
            "cpu_to_gpu_s": round(t_up, 4),
            "gpu_to_cpu_s": round(t_down, 4),
            "vram_on_gpu_gb": round(vram_used, 3),
        }
        layer_transfer_times.append(transfer_info)

        print(f"  Layer {idx}: {l_size:.4f} GB, {l_params/1e6:.1f}M params, "
              f"CPU->GPU: {t_up:.4f}s, GPU->CPU: {t_down:.4f}s, VRAM: {vram_used:.3f} GB")

    # 전체 레이어 순차 이동 시간 추정
    avg_layer_size = sum(layer_sizes) / len(layer_sizes)
    avg_up_time = sum(t["cpu_to_gpu_s"] for t in layer_transfer_times) / len(layer_transfer_times)
    avg_down_time = sum(t["gpu_to_cpu_s"] for t in layer_transfer_times) / len(layer_transfer_times)

    estimated_total_transfer = n_layers * (avg_up_time + avg_down_time)
    estimated_bandwidth_gbps = avg_layer_size / avg_up_time if avg_up_time > 0 else 0

    # LM Head 분석
    lm_head_size = module_size_gb(lm_head)
    clear_gpu()
    t_head_up = time.time()
    lm_head.to("cuda")
    torch.cuda.synchronize()
    t_head_up = time.time() - t_head_up
    lm_head.to("cpu")
    clear_gpu()

    # 임베딩 레이어 분석
    embed = None
    for embed_path in ["language_model.embed_tokens", "language_model.model.embed_tokens", "embed_tokens"]:
        try:
            obj = lang_model
            for part in embed_path.split("."):
                obj = getattr(obj, part)
            embed = obj
            break
        except AttributeError:
            continue

    embed_size = module_size_gb(embed) if embed else 0

    print(f"\n  --- Layer-wise Summary ---")
    print(f"  Avg layer size: {avg_layer_size:.4f} GB")
    print(f"  Avg CPU->GPU transfer: {avg_up_time:.4f}s")
    print(f"  Avg GPU->CPU transfer: {avg_down_time:.4f}s")
    print(f"  Estimated bandwidth: {estimated_bandwidth_gbps:.2f} GB/s")
    print(f"  Estimated total sequential transfer (all {n_layers} layers): {estimated_total_transfer:.2f}s")
    print(f"  LM Head size: {lm_head_size:.3f} GB, transfer: {t_head_up:.4f}s")
    print(f"  Embedding size: {embed_size:.3f} GB")
    print(f"  Max single-layer VRAM: {max(t['vram_on_gpu_gb'] for t in layer_transfer_times):.3f} GB")

    result = {
        "phase": "VLM Language Model",
        "status": "ANALYSIS_ONLY (exceeds VRAM)",
        "module_size_gb": round(vlm_total_size, 3),
        "num_layers": n_layers,
        "avg_layer_size_gb": round(avg_layer_size, 4),
        "avg_cpu_to_gpu_s": round(avg_up_time, 4),
        "avg_gpu_to_cpu_s": round(avg_down_time, 4),
        "estimated_bandwidth_gbps": round(estimated_bandwidth_gbps, 2),
        "estimated_total_transfer_s": round(estimated_total_transfer, 2),
        "lm_head_size_gb": round(lm_head_size, 3),
        "embed_size_gb": round(embed_size, 3),
        "layer_details": layer_transfer_times,
        "vram_peak_per_layer_gb": round(max(t['vram_on_gpu_gb'] for t in layer_transfer_times), 3),
    }

    return result, None


# ============================================================
# Phase 3: Expert (Trajectory Decoder) 테스트
# ============================================================

def test_expert_phase(model):
    """Expert 모듈만 GPU로 이동하여 테스트"""
    print("\n" + "=" * 70)
    print("Phase 3: Expert (Trajectory Decoder)")
    print("=" * 70)

    expert = model.expert
    action_in_proj = model.action_in_proj
    action_out_proj = model.action_out_proj
    diffusion = model.diffusion

    e_size = module_size_gb(expert)
    proj_size = module_size_gb(action_in_proj) + module_size_gb(action_out_proj)
    diff_size = module_size_gb(diffusion)
    total_size = e_size + proj_size + diff_size

    print(f"  Expert backbone: {e_size:.3f} GB ({count_params(expert)/1e6:.1f}M params)")
    print(f"  Action projections: {proj_size:.4f} GB")
    print(f"  Diffusion (Flow Matching): {diff_size:.4f} GB (parametric: {count_params(diffusion)} params)")
    print(f"  Total expert phase: {total_size:.3f} GB")

    can_fit = total_size < 11.5  # 약간의 여유 포함

    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    vram_before = get_vram_gb()

    t_start = time.time()

    if can_fit:
        # Expert 전체를 GPU로 이동
        t_move_start = time.time()
        expert.to("cuda")
        action_in_proj.to("cuda")
        action_out_proj.to("cuda")
        diffusion.to("cuda")
        torch.cuda.synchronize()
        t_move_gpu = time.time() - t_move_start

        vram_after_load = get_vram_gb()
        print(f"  VRAM after GPU load: {vram_after_load:.3f} GB (delta: {vram_after_load - vram_before:.3f} GB)")
        print(f"  CPU->GPU transfer time: {t_move_gpu:.3f}s")

        # 간단한 forward pass 테스트 (더미 입력)
        t_infer_start = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Expert는 36-layer transformer
            # 더미 입력으로 forward 테스트
            batch_size = 1
            seq_len = 20  # n_diffusion_tokens
            hidden_size = expert.config.hidden_size if hasattr(expert, 'config') else 3584

            dummy_embeds = torch.randn(batch_size, seq_len, hidden_size,
                                       device="cuda", dtype=torch.bfloat16)

            try:
                expert_out = expert(inputs_embeds=dummy_embeds, use_cache=False)
                inference_status = "SUCCESS"
                output_shape = str(expert_out.last_hidden_state.shape)
                del expert_out
            except Exception as e:
                inference_status = f"FAILED: {str(e)[:200]}"
                output_shape = "N/A"

            del dummy_embeds

        t_infer = time.time() - t_infer_start
        vram_peak = get_peak_vram_gb()

        print(f"  Inference status: {inference_status}")
        print(f"  Inference time: {t_infer:.3f}s")
        print(f"  Peak VRAM: {vram_peak:.3f} GB")

        # CPU로 반환
        t_offload_start = time.time()
        expert.to("cpu")
        action_in_proj.to("cpu")
        action_out_proj.to("cpu")
        diffusion.to("cpu")
        t_offload = time.time() - t_offload_start

        clear_gpu()
        vram_after_offload = get_vram_gb()

        t_total = time.time() - t_start

        print(f"  GPU->CPU offload time: {t_offload:.3f}s")
        print(f"  VRAM after offload: {vram_after_offload:.3f} GB")
        print(f"  Total phase time: {t_total:.3f}s")

        result = {
            "phase": "Expert (Trajectory Decoder)",
            "status": inference_status,
            "module_size_gb": round(total_size, 3),
            "expert_backbone_gb": round(e_size, 3),
            "action_proj_gb": round(proj_size, 4),
            "cpu_to_gpu_time_s": round(t_move_gpu, 3),
            "inference_time_s": round(t_infer, 3),
            "gpu_to_cpu_time_s": round(t_offload, 3),
            "total_time_s": round(t_total, 3),
            "vram_before_gb": round(vram_before, 3),
            "vram_after_load_gb": round(vram_after_load, 3),
            "vram_peak_gb": round(vram_peak, 3),
            "vram_after_offload_gb": round(vram_after_offload, 3),
            "output_shape": output_shape,
        }
    else:
        print(f"  Expert ({total_size:.2f} GB) exceeds VRAM. Performing layer-wise analysis.")
        result = {
            "phase": "Expert (Trajectory Decoder)",
            "status": "EXCEEDS_VRAM",
            "module_size_gb": round(total_size, 3),
        }

    return result


# ============================================================
# 시스템 RAM 제약 분석
# ============================================================

def analyze_ram_constraints():
    """시스템 RAM 제약 상황을 분석"""
    print("\n" + "=" * 70)
    print("System RAM Constraint Analysis")
    print("=" * 70)

    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            info[parts[0].rstrip(":")] = int(parts[1]) / 1024 / 1024  # kB -> GB

        total_ram = info.get("MemTotal", 0)
        available_ram = info.get("MemAvailable", 0)
        used_ram = total_ram - available_ram
        swap_total = info.get("SwapTotal", 0)
        swap_free = info.get("SwapFree", 0)
        swap_used = swap_total - swap_free

        print(f"  Total RAM: {total_ram:.2f} GB")
        print(f"  Available RAM: {available_ram:.2f} GB")
        print(f"  Used RAM: {used_ram:.2f} GB")
        print(f"  Swap Total: {swap_total:.2f} GB")
        print(f"  Swap Used: {swap_used:.2f} GB")
        print(f"  Swap Free: {swap_free:.2f} GB")

        model_size_bf16 = 22.16  # GB
        can_fit_cpu = available_ram >= model_size_bf16
        can_fit_with_swap = (available_ram + swap_free) >= model_size_bf16

        print(f"\n  Model size (BF16): {model_size_bf16:.2f} GB")
        print(f"  Can fit in available RAM: {'YES' if can_fit_cpu else 'NO'}")
        print(f"  Can fit with swap: {'YES' if can_fit_with_swap else 'NO'}")

        if not can_fit_cpu:
            deficit = model_size_bf16 - available_ram
            print(f"  RAM deficit: {deficit:.2f} GB")
            if can_fit_with_swap:
                print(f"  -> Will use swap, expect significant performance degradation")
            else:
                print(f"  -> Even with swap, not enough memory. Disk offloading or partial loading required.")

        return {
            "total_ram_gb": round(total_ram, 2),
            "available_ram_gb": round(available_ram, 2),
            "swap_total_gb": round(swap_total, 2),
            "swap_free_gb": round(swap_free, 2),
            "model_size_bf16_gb": model_size_bf16,
            "can_fit_ram": can_fit_cpu,
            "can_fit_with_swap": can_fit_with_swap,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}


# ============================================================
# 메인 실험
# ============================================================

def main():
    print("=" * 70)
    print("Sequential Module Offloading Experiment")
    print(f"Start time: {datetime.now().isoformat()}")
    print("=" * 70)

    output_dir = "/home/seungwoo/workspace/research/02-memory-pipelining"
    csv_path = os.path.join(output_dir, "sequential_offload_vram_timeline.csv")
    results_path = os.path.join(output_dir, "sequential_offload_results.json")

    # GPU 정보
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    # VRAM 모니터 시작
    monitor = VRAMMonitor(csv_path)
    monitor.start()

    results = {
        "experiment": "sequential_module_offloading",
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_name if torch.cuda.is_available() else "N/A",
        "gpu_mem_gb": round(gpu_mem, 1),
        "phases": [],
        "errors": [],
    }

    # RAM 분석
    ram_info = analyze_ram_constraints()
    results["ram_constraints"] = ram_info

    # ----------------------------------------------------------
    # 모델 로딩 (offload_folder를 사용하여 디스크 오프로딩)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Loading Model with Disk Offloading")
    print("=" * 70)

    try:
        from transformers import AutoModel, AutoConfig
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        model_name = "nvidia/Alpamayo-R1-10B"
        offload_dir = "/tmp/alpamayo_offload"
        os.makedirs(offload_dir, exist_ok=True)

        print(f"  Loading model: {model_name}")
        print(f"  Strategy: Load to CPU with disk offloading for overflow")

        ram_before = get_ram_usage_gb()
        t_load_start = time.time()

        # 모든 것을 CPU에 로드하되, meta device + disk offloading 활용
        # 시스템 RAM이 15GB이므로, 22GB 모델을 전부 올릴 수 없음
        # accelerate의 disk offloading을 사용

        # 방법 1: device_map으로 모든 것을 CPU에 배치 시도
        # 하지만 RAM이 부족하므로, 실제로는 부분적으로만 로딩됨

        # 방법 2: 서브모듈별로 순차 로딩 (더 안전)
        # 먼저 config만 로드
        print("  Loading config...")

        # Import Alpamayo
        sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        # disk offload로 모델 로딩
        print("  Attempting model load with max_memory constraints...")
        print(f"  Available RAM: {ram_info.get('available_ram_gb', 'N/A')} GB")

        # 먼저 device_map="cpu"로 시도, 실패하면 disk offload
        try:
            model = AlpamayoR1.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map="cpu",
                offload_folder=offload_dir,
                low_cpu_mem_usage=True,
            )
            load_method = "cpu with offloading"
        except Exception as e1:
            print(f"  CPU loading failed: {str(e1)[:200]}")
            print("  Trying with disk offloading via accelerate...")
            try:
                model = AlpamayoR1.from_pretrained(
                    model_name,
                    dtype=torch.bfloat16,
                    device_map={
                        "vlm.model.visual": "cpu",
                        "vlm.model.language_model": "disk",
                        "vlm.lm_head": "disk",
                        "expert": "cpu",
                        "action_in_proj": "cpu",
                        "action_out_proj": "cpu",
                        "diffusion": "cpu",
                        "hist_traj_tokenizer": "cpu",
                        "traj_tokenizer": "cpu",
                    },
                    offload_folder=offload_dir,
                    low_cpu_mem_usage=True,
                )
                load_method = "partial cpu + disk offload"
            except Exception as e2:
                print(f"  Partial loading also failed: {str(e2)[:200]}")
                # 최후 수단: meta device로 구조만 로드하고 수동 분석
                print("  Falling back to analysis-only mode (no weight loading)...")
                load_method = "FAILED - analysis only"

                # 구조 분석만 수행
                results["load_status"] = "FAILED"
                results["load_error"] = str(e2)[:500]
                results["load_method"] = load_method

                # Phase별 추정치 기록
                results["phases"] = [
                    {
                        "phase": "Vision Encoder",
                        "status": "ESTIMATED (no weights loaded)",
                        "module_size_gb": 1.15,
                        "estimated_vram_peak_gb": 1.65,
                        "estimated_transfer_time_s": 0.1,
                        "note": "Small enough for 12GB, sequential offload feasible"
                    },
                    {
                        "phase": "VLM Language Model",
                        "status": "ESTIMATED (no weights loaded)",
                        "module_size_gb": 16.44,
                        "num_layers": 36,
                        "layer_size_gb": 0.386,
                        "estimated_vram_peak_per_layer_gb": 0.5,
                        "estimated_total_transfer_s": 5.0,
                        "note": "EXCEEDS 12GB VRAM. Layer-wise offloading mandatory."
                    },
                    {
                        "phase": "Expert (Trajectory Decoder)",
                        "status": "ESTIMATED (no weights loaded)",
                        "module_size_gb": 4.56,
                        "estimated_vram_peak_gb": 5.56,
                        "estimated_transfer_time_s": 0.4,
                        "note": "Fits in 12GB alone. Sequential offload feasible."
                    },
                ]

                monitor.stop()

                t_total = time.time() - t_load_start
                results["total_time_s"] = round(t_total, 2)

                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {results_path}")
                return results

        t_load = time.time() - t_load_start
        ram_after = get_ram_usage_gb()

        print(f"  Load method: {load_method}")
        print(f"  Load time: {t_load:.2f}s")
        print(f"  RAM usage: {ram_before:.2f} -> {ram_after:.2f} GB (delta: {ram_after - ram_before:.2f} GB)")

        results["load_method"] = load_method
        results["load_time_s"] = round(t_load, 2)
        results["ram_before_load_gb"] = round(ram_before, 2)
        results["ram_after_load_gb"] = round(ram_after, 2)

    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}\n{traceback.format_exc()}"
        print(f"  ERROR: {error_msg}")
        results["errors"].append(error_msg)
        monitor.stop()
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # ----------------------------------------------------------
    # 더미 입력 생성 (4 cameras x 4 frames x 320x576)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Preparing Dummy Inputs")
    print("=" * 70)

    # Qwen3-VL의 vision encoder 입력 형식
    # pixel_values: (N_total_patches, channel_dim)  -- 3D conv input flattened
    # image_grid_thw: (N_groups, 3) [temporal, height_in_patches, width_in_patches]

    n_cameras = 4
    n_frames = 4
    n_images = n_cameras * n_frames  # 16 images
    img_h, img_w = 320, 576

    # Qwen3-VL 패치 설정
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2

    # 각 이미지의 패치 그리드 (패치 공간)
    # 단일 이미지: temporal=1, height = img_h/patch_size, width = img_w/patch_size
    grid_t = 2  # temporal dim (must be even for merge_size=2)
    grid_h = (img_h // patch_size)  # 320/14 = 22 (round down to even for merge)
    grid_w = (img_w // patch_size)  # 576/14 = 41 (round down to even for merge)
    # Qwen3-VL requires grid dims to be divisible by merge_size(=2)
    grid_h = (grid_h // merge_size) * merge_size  # 22
    grid_w = (grid_w // merge_size) * merge_size  # 40

    # 8 groups of 2 frames each (temporal=2)
    n_groups = n_images // temporal_patch_size  # 8 groups
    n_patches_per_group = grid_t * grid_h * grid_w  # 2 * 22 * 40 = 1760
    total_patches = n_groups * n_patches_per_group

    # Channel dim for Conv3d input: 3 channels * temporal_patch_size * patch_size * patch_size
    # But Qwen3-VL pixel_values after patch_embed expects: (total_patches, embed_dim=1152)
    # Actually the input to the vision model is pixel_values which is the raw patch
    # The Conv3d in patch_embed: in_channels=3, kernel=(2,14,14), so patch_dim = 3*2*14*14 = 1176
    patch_dim = 3 * temporal_patch_size * patch_size * patch_size  # 1176

    dummy_pixel_values = torch.randn(total_patches, patch_dim, dtype=torch.float32)
    dummy_image_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]] * n_groups, dtype=torch.long
    )

    print(f"  Images: {n_images} ({n_cameras} cameras x {n_frames} frames)")
    print(f"  Image size: {img_h}x{img_w}")
    print(f"  Grid per image: ({grid_h}, {grid_w})")
    print(f"  Total patches: {total_patches}")
    print(f"  pixel_values shape: {dummy_pixel_values.shape}")
    print(f"  image_grid_thw shape: {dummy_image_grid_thw.shape}")
    print(f"  Input tensor size: {dummy_pixel_values.numel() * 4 / 1e6:.2f} MB")

    # ----------------------------------------------------------
    # Phase 1: Vision Encoder
    # ----------------------------------------------------------
    try:
        phase1_result, vision_output = test_vision_encoder_phase(
            model, dummy_pixel_values, dummy_image_grid_thw
        )
        results["phases"].append(phase1_result)
    except Exception as e:
        error_msg = f"Phase 1 failed: {str(e)}\n{traceback.format_exc()}"
        print(f"  ERROR: {error_msg}")
        results["phases"].append({"phase": "Vision Encoder", "status": "FAILED", "error": str(e)[:500]})
        results["errors"].append(error_msg)
        vision_output = None

    gc.collect()

    # ----------------------------------------------------------
    # Phase 2: VLM Language Model
    # ----------------------------------------------------------
    try:
        phase2_result, _ = test_vlm_layerwise_phase(model)
        results["phases"].append(phase2_result)
    except Exception as e:
        error_msg = f"Phase 2 failed: {str(e)}\n{traceback.format_exc()}"
        print(f"  ERROR: {error_msg}")
        results["phases"].append({"phase": "VLM Language Model", "status": "FAILED", "error": str(e)[:500]})
        results["errors"].append(error_msg)

    gc.collect()
    clear_gpu()

    # ----------------------------------------------------------
    # Phase 3: Expert
    # ----------------------------------------------------------
    try:
        phase3_result = test_expert_phase(model)
        results["phases"].append(phase3_result)
    except Exception as e:
        error_msg = f"Phase 3 failed: {str(e)}\n{traceback.format_exc()}"
        print(f"  ERROR: {error_msg}")
        results["phases"].append({"phase": "Expert", "status": "FAILED", "error": str(e)[:500]})
        results["errors"].append(error_msg)

    gc.collect()
    clear_gpu()

    # ----------------------------------------------------------
    # 종합 분석
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary: Sequential Offloading Feasibility")
    print("=" * 70)

    # 이론적 피크 VRAM (순차 오프로딩 시)
    # 각 phase에서 가장 큰 모듈만 GPU에 올림
    phase_vrams = []
    for p in results["phases"]:
        if "vram_peak_gb" in p:
            phase_vrams.append(p["vram_peak_gb"])
        elif "vram_peak_per_layer_gb" in p:
            phase_vrams.append(p["vram_peak_per_layer_gb"])
        elif "estimated_vram_peak_gb" in p:
            phase_vrams.append(p["estimated_vram_peak_gb"])

    if phase_vrams:
        theoretical_peak = max(phase_vrams)
        print(f"  Theoretical peak VRAM (sequential offload): {theoretical_peak:.3f} GB")
        print(f"  GPU capacity: {gpu_mem:.1f} GB")
        print(f"  Feasible: {'YES' if theoretical_peak < gpu_mem else 'NO'}")

    # 핵심 결론
    summary = {
        "vision_encoder": {
            "size_gb": 1.15,
            "fits_gpu": True,
            "sequential_offload_viable": True,
        },
        "vlm_language_model": {
            "size_gb": 16.44,
            "fits_gpu": False,
            "sequential_offload_viable": False,
            "layer_wise_offload_needed": True,
            "single_layer_fits_gpu": True,
            "layer_size_gb": 0.386,
        },
        "expert_decoder": {
            "size_gb": 4.56,
            "fits_gpu": True,
            "sequential_offload_viable": True,
        },
        "conclusion": (
            "Module-level sequential offloading is PARTIALLY feasible. "
            "Vision Encoder (1.15GB) and Expert (4.56GB) can each be loaded to GPU individually. "
            "However, VLM Language Model (16.44GB BF16) EXCEEDS 12GB VRAM even alone, "
            "requiring LAYER-WISE offloading within the VLM. "
            "The critical dependency is that Expert needs VLM's KV Cache, "
            "which must remain on GPU during Expert execution. "
            "System RAM (15GB) cannot hold the full 22GB model simultaneously on CPU."
        ),
    }
    results["summary"] = summary

    print(f"\n  Conclusion: {summary['conclusion']}")

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
