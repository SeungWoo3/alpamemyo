#!/usr/bin/env python3
"""Part 2b: Manual On-Demand Offloading Experiment

Since total system RAM (15GB) cannot hold the full model (22GB BF16) on CPU,
we use accelerate's disk offloading to complement CPU memory.

Strategy:
1. Load model with device_map that puts some on GPU, rest on CPU+disk
2. Measure per-stage VRAM by using accelerate's dispatch
3. Test actual inference with sequential offloading

Alternative approach: Load with max_memory constraints to simulate
on-demand layering behavior.
"""

import sys
import os
import time
import csv
import json
import threading
import gc

sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")

import torch
import numpy as np

OUTPUT_DIR = "/home/seungwoo/workspace/research/01-on-demand-layering"
OFFLOAD_DIR = "/tmp/alpamayo_offload"


# ─── VRAM Monitor ────────────────────────────────────────────────────────────

class VRAMMonitor:
    """Monitor GPU VRAM usage at regular intervals."""

    def __init__(self, interval=0.5, device=0):
        self.interval = interval
        self.device = device
        self.records = []
        self._stop = threading.Event()
        self._thread = None
        self._phase = "init"

    def set_phase(self, phase):
        self._phase = phase

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _monitor(self):
        t0 = time.time()
        while not self._stop.is_set():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            self.records.append({
                "time": round(time.time() - t0, 2),
                "allocated_gb": round(allocated, 3),
                "reserved_gb": round(reserved, 3),
                "phase": self._phase,
            })
            self._stop.wait(self.interval)

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "allocated_gb", "reserved_gb", "phase"])
            writer.writeheader()
            writer.writerows(self.records)

    def peak_allocated(self):
        if not self.records:
            return 0
        return max(r["allocated_gb"] for r in self.records)


def get_module_size_gb(module):
    """Get size of a module's parameters in GB."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return total / 1e9


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Part 2b: Manual On-Demand Offloading Experiment")
    print("=" * 80)
    print(f"    System RAM: ~15 GB, Model size: ~22 GB (BF16)")
    print(f"    Strategy: Use accelerate disk offloading + per-stage GPU movement")

    os.makedirs(OFFLOAD_DIR, exist_ok=True)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper

    # ── Approach 1: Measure per-submodule sizes using meta device ──
    print("\n[1/5] Estimating per-submodule sizes using layer_analysis.json...")

    with open(os.path.join(OUTPUT_DIR, "layer_analysis.json")) as f:
        layer_data = json.load(f)

    semantic = layer_data["semantic_groups"]
    print("    Component sizes (BF16 = same as FP16 element count, 2 bytes/param):")
    for name, info in semantic.items():
        gb = info["memory_bytes"] / 1e9
        if gb > 0.001:
            print(f"      {name:<50} {gb:.2f} GB")

    vision_enc_gb = semantic["Vision Encoder (vlm.visual)"]["memory_bytes"] / 1e9
    vlm_lm_gb = semantic["VLM Language Model (vlm.model - visual)"]["memory_bytes"] / 1e9
    vlm_head_gb = semantic["VLM LM Head"]["memory_bytes"] / 1e9
    expert_gb = semantic["Expert (Trajectory Decoder Backbone)"]["memory_bytes"] / 1e9

    # ── Approach 2: Load model with max_memory constraint ──
    # This forces accelerate to use CPU offloading with hooks
    print("\n[2/5] Loading model with max_memory constraint...")
    print(f"    max_memory: GPU=10GB, CPU=10GB (disk offload for remainder)")

    t_load_start = time.time()

    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "10GiB", "cpu": "10GiB"},
        offload_folder=OFFLOAD_DIR,
        low_cpu_mem_usage=True,
    )
    model.eval()

    t_load_end = time.time()
    load_time = t_load_end - t_load_start
    print(f"    Model loaded in {load_time:.1f}s")

    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"    VRAM after load: {vram_after_load:.2f} GB")

    # Analyze where parameters ended up
    gpu_params = 0
    cpu_params = 0
    disk_params = 0  # meta device = offloaded to disk
    for name, param in model.named_parameters():
        n = param.numel()
        dev = str(param.device)
        if "cuda" in dev:
            gpu_params += n
        elif "meta" in dev:
            disk_params += n
        else:
            cpu_params += n

    total_p = gpu_params + cpu_params + disk_params
    print(f"    GPU: {gpu_params/1e9:.2f}B ({gpu_params/total_p*100:.0f}%)")
    print(f"    CPU: {cpu_params/1e9:.2f}B ({cpu_params/total_p*100:.0f}%)")
    print(f"    Disk: {disk_params/1e9:.2f}B ({disk_params/total_p*100:.0f}%)")

    # ── Approach 3: Run inference with this setup ──
    print("\n[3/5] Preparing dummy input...")
    dummy_images = torch.randn(16, 3, 320, 576)
    tokenizer = model.tokenizer
    processor = helper.get_processor(tokenizer)
    messages = helper.create_message(dummy_images)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    ego_history_xyz = torch.zeros(1, 1, 4, 3)
    ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 4, 3, 3).clone()

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    print("\n[4/5] Running inference with VRAM monitoring...")
    monitor = VRAMMonitor(interval=0.5)
    monitor.start()
    monitor.set_phase("inference")

    torch.cuda.reset_peak_memory_stats()
    t_infer_start = time.time()
    inference_success = False

    try:
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                )
        inference_success = True
        print(f"    Inference succeeded!")
    except torch.cuda.OutOfMemoryError as e:
        print(f"    OOM: {e}")
    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")

    t_infer_end = time.time()
    monitor.stop()

    inference_time = t_infer_end - t_infer_start
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"    Inference time: {inference_time:.2f}s")
    print(f"    Peak VRAM (PyTorch): {peak_vram:.2f} GB")
    print(f"    Peak VRAM (Monitor): {monitor.peak_allocated():.2f} GB")
    print(f"    Inference success: {inference_success}")

    # Save timeline
    csv_path = os.path.join(OUTPUT_DIR, "manual_offload_vram_timeline.csv")
    monitor.save_csv(csv_path)
    print(f"    Timeline saved to: {csv_path}")

    # ── Step 5: On-Demand Layering analysis ──
    print("\n[5/5] On-Demand Layering feasibility analysis:")
    print("=" * 80)

    print("\n    === Scenario A: Pure On-Demand Layering (per-layer GPU) ===")
    # Per-layer size for expert: 63.31M params = ~126.62 MB
    expert_layer_gb = 63.31e6 * 2 / 1e9  # BF16
    vlm_layer_gb = 192.9e6 * 2 / 1e9
    print(f"    VLM layer size: {vlm_layer_gb*1000:.0f} MB ({vlm_layer_gb:.3f} GB)")
    print(f"    Expert layer size: {expert_layer_gb*1000:.0f} MB ({expert_layer_gb:.3f} GB)")
    print(f"    Vision Encoder total: {vision_enc_gb:.2f} GB")
    print(f"    Double-buffer VRAM (largest layer x 2 + activation): ~{vlm_layer_gb*2 + 1:.2f} GB")
    print(f"    => Theoretical minimum VRAM: ~{vlm_layer_gb*2 + 1:.1f} GB")
    print(f"    => Memory reduction: {(1 - (vlm_layer_gb*2 + 1) / 22.16) * 100:.1f}%")

    print("\n    === Scenario B: Module-level On-Demand (recommended) ===")
    max_module_vram = max(vision_enc_gb, vlm_lm_gb + vlm_head_gb, expert_gb)
    print(f"    Vision Encoder: {vision_enc_gb:.2f} GB")
    print(f"    VLM LM + Head: {vlm_lm_gb + vlm_head_gb:.2f} GB")
    print(f"    Expert: {expert_gb:.2f} GB")
    print(f"    Peak VRAM (largest module + activation): ~{max_module_vram + 2:.2f} GB")
    print(f"    => This exceeds 12 GB! VLM alone = {vlm_lm_gb + vlm_head_gb:.2f} GB")
    print(f"    => Need quantization or layer-level offloading for VLM")

    print("\n    === Scenario C: Hybrid (VLM quantized INT4 + sequential offload) ===")
    vlm_int4_gb = (vlm_lm_gb + vlm_head_gb) / 4  # ~4x compression
    expert_int4_gb = expert_gb / 4
    print(f"    VLM LM + Head (INT4): ~{vlm_int4_gb:.2f} GB")
    print(f"    Expert (INT4): ~{expert_int4_gb:.2f} GB")
    print(f"    Peak VRAM (VLM INT4 + activation): ~{vlm_int4_gb + 2:.2f} GB")
    print(f"    => Fits in 12 GB!")

    # ── Save results ──
    results = {
        "experiment": "manual_offload",
        "model": "nvidia/Alpamayo-R1-10B",
        "dtype": "bfloat16",
        "gpu": "RTX 3080 Ti (12GB)",
        "system_ram_gb": 15,
        "load_time_seconds": round(load_time, 2),
        "vram_after_load_gb": round(vram_after_load, 2),
        "inference": {
            "success": inference_success,
            "inference_time_s": round(inference_time, 2),
            "peak_vram_pytorch_gb": round(peak_vram, 2),
            "peak_vram_monitor_gb": round(monitor.peak_allocated(), 2),
        },
        "device_distribution": {
            "gpu_params_B": round(gpu_params / 1e9, 2),
            "cpu_params_B": round(cpu_params / 1e9, 2),
            "disk_params_B": round(disk_params / 1e9, 2),
        },
        "per_stage_vram_gb": {
            "vision_encoder": round(vision_enc_gb, 2),
            "vlm_lm_plus_head": round(vlm_lm_gb + vlm_head_gb, 2),
            "expert": round(expert_gb, 2),
        },
        "on_demand_layering_analysis": {
            "vlm_per_layer_mb": round(vlm_layer_gb * 1000, 1),
            "expert_per_layer_mb": round(expert_layer_gb * 1000, 1),
            "scenario_a_theoretical_min_vram_gb": round(vlm_layer_gb * 2 + 1, 2),
            "scenario_a_memory_reduction_pct": round((1 - (vlm_layer_gb * 2 + 1) / 22.16) * 100, 1),
            "scenario_b_peak_vram_gb": round(max_module_vram + 2, 2),
            "scenario_c_vlm_int4_vram_gb": round(vlm_int4_gb + 2, 2),
        },
    }

    results_path = os.path.join(OUTPUT_DIR, "manual_offload_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n    Results saved to: {results_path}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Clean up disk offload
    import shutil
    if os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
