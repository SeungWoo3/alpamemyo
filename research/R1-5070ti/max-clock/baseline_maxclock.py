import torch
import numpy as np
import time
import threading
import json
import subprocess

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

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

def get_gpu_clocks():
    """현재 GPU 클럭 정보 조회"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw,power.limit"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return "N/A"

def main():
    results = {}

    # 클럭 정보 기록
    clock_info_before = get_gpu_clocks()
    print(f"GPU Clocks (before): {clock_info_before}")
    results["clock_info_before"] = clock_info_before

    monitor = VRAMMonitor(interval=0.1)
    monitor.start()

    # Phase 1: 데이터 로드
    print("Phase 1: Loading dataset...")
    t0 = time.time()
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    t_data = time.time() - t0
    print(f"  Dataset loaded: {t_data:.2f}s")
    results["data_load_time"] = t_data

    # Phase 2: 모델 로드 (device_map="auto")
    print("Phase 2: Loading model with device_map='auto'...")
    model_load_start = time.time() - monitor.t0
    t0 = time.time()
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16, device_map="auto")
    t_model = time.time() - t0
    model_load_end = time.time() - monitor.t0
    print(f"  Model loaded: {t_model:.2f}s")
    results["model_load_time"] = t_model
    results["model_load_start_t"] = model_load_start
    results["model_load_end_t"] = model_load_end
    results["vram_after_load"] = torch.cuda.memory_allocated() / 1024**3

    # 클럭 정보 (로드 후)
    clock_info_after_load = get_gpu_clocks()
    print(f"GPU Clocks (after load): {clock_info_after_load}")
    results["clock_info_after_load"] = clock_info_after_load

    # Device map 요약
    dm = model.hf_device_map
    cuda_count = sum(1 for v in dm.values() if str(v) == '0')
    cpu_count = sum(1 for v in dm.values() if str(v) == 'cpu')
    results["device_placement"] = {"cuda": cuda_count, "cpu": cpu_count}

    # Phase 3: 입력 준비
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    first_device = next(model.parameters()).device
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, first_device)

    # Phase 4: 추론
    print("Phase 3: Running inference...")
    inference_start = time.time() - monitor.t0
    torch.cuda.manual_seed_all(42)
    t0 = time.time()
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
    inference_end = time.time() - monitor.t0
    print(f"  Inference done: {t_infer:.2f}s")

    # 클럭 정보 (추론 직후)
    clock_info_after_infer = get_gpu_clocks()
    print(f"GPU Clocks (after inference): {clock_info_after_infer}")
    results["clock_info_after_inference"] = clock_info_after_infer

    results["inference_time"] = t_infer
    results["inference_start_t"] = inference_start
    results["inference_end_t"] = inference_end

    # Phase 5: 결과 계산
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())
    print(f"  minADE: {min_ade:.4f}m")
    results["minADE"] = min_ade
    results["cot"] = str(extra["cot"][0])

    monitor.stop()

    results["peak_vram_allocated"] = max(monitor.allocated)
    results["peak_vram_reserved"] = max(monitor.reserved)
    results["gpu"] = "RTX 5070 Ti 16GB"
    results["method"] = "device_map=auto (max clock)"
    results["total_time"] = time.time() - (monitor.t0)

    vram_data = monitor.to_dict()

    print(f"\n=== Results ===")
    print(f"  Model load: {results['model_load_time']:.2f}s")
    print(f"  Inference: {results['inference_time']:.2f}s")
    print(f"  Peak VRAM: {results['peak_vram_allocated']:.2f} GB")
    print(f"  minADE: {results['minADE']:.4f}m")

    with open("/home/avees/workspace/baseline_maxclock_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open("/home/avees/workspace/baseline_maxclock_vram_timeline.json", "w") as f:
        json.dump(vram_data, f)

    print("Results saved.")

if __name__ == "__main__":
    main()
