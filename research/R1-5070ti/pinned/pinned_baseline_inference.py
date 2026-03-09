"""
5070ti Pinned 베이스라인 실험
- 바닐라 방식 + pin_memory()
- CPU에서 모델 로드 → 전체 파라미터 pin_memory() → .to("cuda")
- Pinned memory DMA로 H2D 전송 속도 향상 기대
- 21.52GB > 16GB VRAM → CUDA Unified Memory 자동 스왑 (or OOM)
"""
import sys
import os
import time
import json
import gc
import threading
import subprocess

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, "/home/avees/workspace/alpamayo/src")

import torch

RESULTS_DIR = "/home/avees/workspace/research/R1-5070ti/pinned"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_ID = "nvidia/Alpamayo-R1-10B"


class VRAMMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.timestamps = []
        self.allocated = []
        self.reserved = []
        self.running = False
        self.t0 = None

    def start(self):
        self.t0 = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _monitor(self):
        while self.running:
            t = time.time() - self.t0
            alloc = torch.cuda.memory_allocated() / 1024**3
            resv = torch.cuda.memory_reserved() / 1024**3
            self.timestamps.append(t)
            self.allocated.append(alloc)
            self.reserved.append(resv)
            time.sleep(self.interval)

    def to_dict(self):
        return {
            "timestamps": self.timestamps,
            "allocated_gb": self.allocated,
            "reserved_gb": self.reserved,
        }


def pin_all_parameters(model):
    """모든 CPU 파라미터를 pinned memory로 변환 (메모리 절약: 원본 즉시 해제)"""
    pinned_count = 0
    pinned_bytes = 0
    for name, param in model.named_parameters():
        if param.device.type == "cpu":
            data = param.data
            if not data.is_contiguous():
                data = data.contiguous()
            if not data.is_pinned():
                pinned = data.pin_memory()
                param.data = torch.empty(0)  # 원본 해제 (RAM 절약)
                param.data = pinned
                del data
                pinned_count += 1
                pinned_bytes += pinned.nelement() * pinned.element_size()
    for name, buf in model.named_buffers():
        if buf.device.type == "cpu" and not buf.is_pinned():
            pinned = buf.data.contiguous().pin_memory()
            buf.data = torch.empty(0)
            buf.data = pinned
    gc.collect()
    return pinned_count, pinned_bytes


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def main():
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    config_name = "pinned_vanilla_baseline"
    print("=" * 60)
    print("5070ti Pinned 베이스라인 (CPU 로드 → pin_memory → .to('cuda'))")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram:.1f} GB")
    print(f"GPU Info: {get_gpu_info()}")

    results = {
        "config": config_name,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gb": round(total_vram, 2),
        "method": "CPU load → pin_memory() → .to('cuda')",
    }
    monitor = VRAMMonitor(interval=0.1)
    monitor.start()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t_total = time.time()

    # Phase 1: 데이터 로드
    print("\n[Phase 1] Loading dataset...")
    t0 = time.time()
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    t_data = time.time() - t0
    print(f"  Dataset loaded: {t_data:.2f}s")
    results["data_load_time"] = round(t_data, 2)

    # Phase 2: 모델 CPU 로드
    print("\n[Phase 2] Loading model to CPU...")
    model_load_start = time.time() - monitor.t0
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    t_cpu_load = time.time() - t0
    print(f"  CPU load: {t_cpu_load:.2f}s")
    results["cpu_load_time"] = round(t_cpu_load, 2)

    # Phase 3: pin_memory() 적용
    print("\n[Phase 3] Pinning all parameters...")
    t0 = time.time()
    pinned_count, pinned_bytes = pin_all_parameters(model)
    t_pin = time.time() - t0
    print(f"  Pinned {pinned_count} params ({pinned_bytes / 1024**3:.2f} GB)")
    print(f"  Pin time: {t_pin:.2f}s")
    results["pin_count"] = pinned_count
    results["pin_bytes_gb"] = round(pinned_bytes / 1024**3, 2)
    results["pin_time"] = round(t_pin, 2)

    # Phase 4: .to("cuda") — Unified Memory 스왑 or OOM
    print("\n[Phase 4] Moving to CUDA (.to('cuda'))...")
    print("  (21.52GB > 16GB VRAM → Unified Memory 스왑 or OOM)")
    t0 = time.time()
    try:
        model = model.to("cuda")
        t_to_cuda = time.time() - t0
        print(f"  .to('cuda') done: {t_to_cuda:.2f}s")
        results["to_cuda_time"] = round(t_to_cuda, 2)
        results["to_cuda_error"] = None
    except torch.OutOfMemoryError as e:
        t_to_cuda = time.time() - t0
        print(f"  [OOM] {e}")
        results["to_cuda_time"] = round(t_to_cuda, 2)
        results["to_cuda_error"] = str(e)
        monitor.stop()
        results["total_time"] = round(time.time() - t_total, 2)
        results["peak_vram_allocated"] = max(monitor.allocated) if monitor.allocated else 0
        results["peak_vram_reserved"] = max(monitor.reserved) if monitor.reserved else 0
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(RESULTS_DIR, "vram_timeline.json"), "w") as f:
            json.dump(monitor.to_dict(), f)
        print(f"\nOOM results saved to {RESULTS_DIR}")
        return

    results["model_load_start_t"] = model_load_start
    results["model_load_end_t"] = time.time() - monitor.t0
    results["model_load_time"] = round(t_cpu_load + t_pin + t_to_cuda, 2)
    results["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)

    # Phase 5: 입력 준비
    print("\n[Phase 5] Preparing inputs...")
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # Phase 6: 추론
    print("\n[Phase 6] Running inference...")
    print(f"  GPU Info: {get_gpu_info()}")
    torch.cuda.reset_peak_memory_stats()
    inference_start = time.time() - monitor.t0
    t0 = time.time()

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    t_infer = time.time() - t0
    print(f"  Inference done: {t_infer:.2f}s")

    results["inference_time"] = round(t_infer, 2)
    results["inference_start_t"] = inference_start
    results["inference_end_t"] = time.time() - monitor.t0
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)

    # Phase 7: 정확도
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())
    print(f"  minADE: {min_ade:.4f}m")
    results["minADE"] = round(min_ade, 4)
    results["cot"] = str(extra["cot"][0])
    results["error"] = None

    results["gpu_info_after"] = get_gpu_info()
    results["total_time"] = round(time.time() - t_total, 2)

    monitor.stop()
    results["peak_vram_allocated"] = max(monitor.allocated) if monitor.allocated else 0
    results["peak_vram_reserved"] = max(monitor.reserved) if monitor.reserved else 0

    vram_data = monitor.to_dict()

    # 결과 출력
    print(f"\n{'='*60}")
    print("RESULTS (Pinned Vanilla Baseline)")
    print(f"{'='*60}")
    print(f"  CPU load: {results['cpu_load_time']}s")
    print(f"  Pin time: {results['pin_time']}s")
    print(f"  .to(cuda): {results['to_cuda_time']}s")
    print(f"  Inference: {results['inference_time']}s")
    print(f"  Peak VRAM: {results['peak_vram_gb']} GB")
    print(f"  minADE: {results['minADE']}m")
    print(f"{'='*60}")

    # 저장
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "vram_timeline.json"), "w") as f:
        json.dump(vram_data, f)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
