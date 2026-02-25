#!/usr/bin/env python3
"""Part 2a: device_map="auto" experiment

Loads the Alpamayo R1 model with device_map="auto" and measures:
- Which layers are placed on GPU vs CPU
- VRAM usage over time (0.5s interval CSV)
- Inference time with dummy input
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

# ─── VRAM Monitor ────────────────────────────────────────────────────────────

class VRAMMonitor:
    """Monitor GPU VRAM usage at regular intervals."""

    def __init__(self, interval=0.5, device=0):
        self.interval = interval
        self.device = device
        self.records = []
        self._stop = threading.Event()
        self._thread = None

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
            })
            self._stop.wait(self.interval)

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "allocated_gb", "reserved_gb"])
            writer.writeheader()
            writer.writerows(self.records)

    def peak_allocated(self):
        if not self.records:
            return 0
        return max(r["allocated_gb"] for r in self.records)

    def peak_reserved(self):
        if not self.records:
            return 0
        return max(r["reserved_gb"] for r in self.records)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Part 2a: device_map='auto' Experiment")
    print("=" * 80)

    # Clear GPU
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # ── Step 1: Load model ──
    print("\n[1/4] Loading model with device_map='auto'...")
    t_load_start = time.time()

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    t_load_end = time.time()
    load_time = t_load_end - t_load_start
    print(f"    Model loaded in {load_time:.1f}s")

    # ── Step 2: Analyze device placement ──
    print("\n[2/4] Analyzing device placement...")
    device_placement = {}
    gpu_params = 0
    cpu_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        device = str(param.device)
        total_params += param.numel()
        if "cuda" in device:
            gpu_params += param.numel()
        else:
            cpu_params += param.numel()

        # Get top-level module name
        top_module = name.split(".")[0]
        if top_module not in device_placement:
            device_placement[top_module] = {"gpu": 0, "cpu": 0}
        if "cuda" in device:
            device_placement[top_module]["gpu"] += param.numel()
        else:
            device_placement[top_module]["cpu"] += param.numel()

    print(f"\n    GPU parameters: {gpu_params/1e9:.2f}B ({gpu_params/total_params*100:.1f}%)")
    print(f"    CPU parameters: {cpu_params/1e9:.2f}B ({cpu_params/total_params*100:.1f}%)")
    print(f"\n    Device placement by top-level module:")
    for mod_name, counts in device_placement.items():
        total_mod = counts["gpu"] + counts["cpu"]
        if total_mod > 0:
            gpu_pct = counts["gpu"] / total_mod * 100
            print(f"      {mod_name:<25} GPU: {counts['gpu']/1e9:.2f}B ({gpu_pct:.0f}%)  CPU: {counts['cpu']/1e9:.2f}B ({100-gpu_pct:.0f}%)")

    # More detailed: per sub-module for vlm
    print("\n    Detailed VLM device placement:")
    vlm_layer_devices = {}
    for name, param in model.named_parameters():
        if name.startswith("vlm."):
            parts = name.split(".")
            # Get meaningful sub-module (e.g., vlm.model.language_model.layers.0)
            if len(parts) >= 5 and parts[2] == "language_model" and parts[3] == "layers":
                key = f"vlm.model.language_model.layers.{parts[4]}"
            elif len(parts) >= 3 and parts[1] == "model" and parts[2] == "visual":
                key = "vlm.model.visual"
            elif len(parts) >= 2 and parts[1] == "lm_head":
                key = "vlm.lm_head"
            else:
                key = ".".join(parts[:3])

            if key not in vlm_layer_devices:
                vlm_layer_devices[key] = {"gpu": 0, "cpu": 0}
            if "cuda" in str(param.device):
                vlm_layer_devices[key]["gpu"] += param.numel()
            else:
                vlm_layer_devices[key]["cpu"] += param.numel()

    for key in sorted(vlm_layer_devices.keys()):
        counts = vlm_layer_devices[key]
        total_k = counts["gpu"] + counts["cpu"]
        if total_k > 0:
            device_str = "GPU" if counts["gpu"] > counts["cpu"] else "CPU"
            print(f"      {key:<50} -> {device_str}  ({total_k/1e6:.1f}M params)")

    # Expert layer device placement
    print("\n    Detailed Expert device placement:")
    expert_layer_devices = {}
    for name, param in model.named_parameters():
        if name.startswith("expert."):
            parts = name.split(".")
            if len(parts) >= 3 and parts[1] == "layers":
                key = f"expert.layers.{parts[2]}"
            else:
                key = ".".join(parts[:2])

            if key not in expert_layer_devices:
                expert_layer_devices[key] = {"gpu": 0, "cpu": 0}
            if "cuda" in str(param.device):
                expert_layer_devices[key]["gpu"] += param.numel()
            else:
                expert_layer_devices[key]["cpu"] += param.numel()

    for key in sorted(expert_layer_devices.keys(), key=lambda x: (x.split(".")[0], int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else -1)):
        counts = expert_layer_devices[key]
        total_k = counts["gpu"] + counts["cpu"]
        if total_k > 0:
            device_str = "GPU" if counts["gpu"] > counts["cpu"] else "CPU"
            print(f"      {key:<50} -> {device_str}  ({total_k/1e6:.1f}M params)")

    # ── Step 3: Measure VRAM after loading ──
    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"\n    VRAM after model load: {vram_after_load:.2f} GB")

    # ── Step 4: Dummy inference with VRAM monitoring ──
    print("\n[3/4] Running dummy inference with VRAM monitoring...")

    # Create dummy input (4 cameras, 4 frames, 320x576)
    # For device_map auto, inputs go to the device of the first parameter
    from alpamayo_r1 import helper

    # Create dummy images (4 cameras * 4 frames = 16 images)
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

    # Dummy trajectory data
    ego_history_xyz = torch.zeros(1, 1, 4, 3)
    ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 4, 3, 3).clone()

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }

    # Move inputs to appropriate device
    model_inputs = helper.to_device(model_inputs, "cuda")

    # Start VRAM monitor
    monitor = VRAMMonitor(interval=0.5)
    monitor.start()

    torch.cuda.reset_peak_memory_stats()
    t_infer_start = time.time()

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
    except Exception as e:
        print(f"    Inference failed: {e}")
        inference_success = False

    t_infer_end = time.time()
    monitor.stop()

    inference_time = t_infer_end - t_infer_start
    peak_vram_pytorch = torch.cuda.max_memory_allocated() / 1e9
    peak_vram_monitor = monitor.peak_allocated()

    print(f"\n    Inference time: {inference_time:.2f}s")
    print(f"    Peak VRAM (PyTorch): {peak_vram_pytorch:.2f} GB")
    print(f"    Peak VRAM (Monitor): {peak_vram_monitor:.2f} GB")
    print(f"    Inference success: {inference_success}")

    # Save VRAM timeline CSV
    csv_path = os.path.join(OUTPUT_DIR, "device_map_vram_timeline.csv")
    monitor.save_csv(csv_path)
    print(f"    VRAM timeline saved to: {csv_path}")

    # ── Step 5: Save results ──
    print("\n[4/4] Saving results...")
    results = {
        "experiment": "device_map_auto",
        "model": "nvidia/Alpamayo-R1-10B",
        "dtype": "bfloat16",
        "gpu": "RTX 3080 Ti (12GB)",
        "load_time_seconds": round(load_time, 2),
        "inference_time_seconds": round(inference_time, 2),
        "total_time_seconds": round(load_time + inference_time, 2),
        "vram_after_load_gb": round(vram_after_load, 2),
        "peak_vram_pytorch_gb": round(peak_vram_pytorch, 2),
        "peak_vram_monitor_gb": round(peak_vram_monitor, 2),
        "inference_success": inference_success,
        "gpu_params_billion": round(gpu_params / 1e9, 2),
        "cpu_params_billion": round(cpu_params / 1e9, 2),
        "device_placement_summary": {
            mod: {"gpu_params_M": round(c["gpu"]/1e6, 1), "cpu_params_M": round(c["cpu"]/1e6, 1)}
            for mod, c in device_placement.items()
        },
        "n_vram_samples": len(monitor.records),
    }

    results_path = os.path.join(OUTPUT_DIR, "device_map_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Results saved to: {results_path}")

    # Cleanup
    del model, model_inputs
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
