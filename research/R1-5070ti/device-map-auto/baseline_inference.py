"""
5070ti 베이스라인 추론 실험
- device_map="auto": Transformers가 레이어를 GPU/CPU에 자동 배치
- 21.52GB 모델, 16GB VRAM → 일부 레이어 CPU 오프로드 예상
- 측정 항목: 모델 로딩 시간, 추론 시간, VRAM 사용량 시계열
"""
import torch
import numpy as np
import time
import threading
import json

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


class VRAMMonitor:
    """VRAM 사용량 시계열 모니터링"""
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


def get_device_map_summary(model):
    """모델 레이어별 device 배치 요약"""
    device_count = {}
    for name, param in model.named_parameters():
        dev = str(param.device)
        device_count[dev] = device_count.get(dev, 0) + 1
    return device_count


def main():
    print("=" * 60)
    print("5070ti 베이스라인 추론 실험 (device_map='auto')")
    print("=" * 60)

    results = {
        "method": "device_map=auto",
        "gpu": "RTX 5070 Ti 16GB",
        "note": ".to('cuda') 방식은 OOM (21.52GB > 16GB VRAM)"
    }
    monitor = VRAMMonitor(interval=0.1)
    monitor.start()

    # Phase 1: 데이터 로드
    print("\nPhase 1: Loading dataset...")
    t0 = time.time()
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    t_data = time.time() - t0
    print(f"  Dataset loaded: {t_data:.2f}s")
    results["data_load_time"] = t_data

    # Phase 2: 모델 로드 (device_map="auto")
    print("\nPhase 2: Loading model with device_map='auto'...")
    t_model_start = time.time() - monitor.t0  # 모니터 기준 타임스탬프
    t0 = time.time()
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    t_model = time.time() - t0
    print(f"  Model loaded: {t_model:.2f}s")
    results["model_load_time"] = t_model
    results["model_load_start_t"] = t_model_start
    results["model_load_end_t"] = t_model_start + t_model
    results["vram_after_load_allocated"] = torch.cuda.memory_allocated() / 1024**3
    results["vram_after_load_reserved"] = torch.cuda.memory_reserved() / 1024**3

    # 레이어 배치 요약
    dev_summary = get_device_map_summary(model)
    print(f"  Device placement: {dev_summary}")
    results["device_placement"] = dev_summary

    # Phase 3: 입력 준비
    print("\nPhase 3: Preparing inputs...")
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    # device_map="auto" 시 입력은 첫 번째 레이어 device로
    first_device = next(model.parameters()).device
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, first_device)

    # Phase 4: 추론
    print("\nPhase 4: Running inference...")
    t_infer_start = time.time() - monitor.t0
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
    print(f"  Inference done: {t_infer:.2f}s")
    results["inference_time"] = t_infer
    results["inference_start_t"] = t_infer_start
    results["inference_end_t"] = t_infer_start + t_infer

    # Phase 5: 결과 계산
    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())
    print(f"  minADE: {min_ade:.4f}m")
    results["minADE"] = min_ade
    results["cot"] = extra["cot"][0]

    monitor.stop()

    # Peak VRAM
    results["peak_vram_allocated"] = max(monitor.allocated)
    results["peak_vram_reserved"] = max(monitor.reserved)
    results["total_time"] = results["data_load_time"] + results["model_load_time"] + results["inference_time"]

    print(f"\n{'=' * 60}")
    print(f"=== Results ===")
    print(f"  Data load:    {results['data_load_time']:.2f}s")
    print(f"  Model load:   {results['model_load_time']:.2f}s")
    print(f"  Inference:    {results['inference_time']:.2f}s")
    print(f"  Total:        {results['total_time']:.2f}s")
    print(f"  Peak VRAM (allocated): {results['peak_vram_allocated']:.2f} GB")
    print(f"  Peak VRAM (reserved):  {results['peak_vram_reserved']:.2f} GB")
    print(f"  minADE: {results['minADE']:.4f}m")
    print(f"  Device: {dev_summary}")
    print(f"{'=' * 60}")

    # JSON 저장
    with open("/home/avees/workspace/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    vram_data = monitor.to_dict()
    with open("/home/avees/workspace/baseline_vram_timeline.json", "w") as f:
        json.dump(vram_data, f)

    print("Results saved to /home/avees/workspace/")


if __name__ == "__main__":
    main()
