"""
4-bit Alpamayo VRAM 제한별 성능 비교 실험

VRAM을 dummy 텐서로 미리 점유하여 사용 가능한 VRAM을 제한하고,
그 상태에서 4-bit 모델 추론 성능을 비교한다.

실험 조건:
- 12GB (baseline): dummy 없음
- 10GB: 2GB dummy 점유
- 8GB: 4GB dummy 점유
- 6GB: 6GB dummy 점유
"""
import sys
import os
import json
import time
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alpamayo", "src"))

import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "dwko/Alpamayo-R1-10B-4bit"

# VRAM limits to test (in GB of available VRAM)
VRAM_LIMITS = [12, 10, 8, 6]

def create_dummy_data():
    """더미 입력 데이터 생성"""
    num_cameras, num_frames = 4, 4
    H, W = 320, 576
    image_frames = torch.randint(0, 256, (num_cameras, num_frames, 3, H, W), dtype=torch.uint8)
    ego_history_xyz = torch.zeros(1, 1, 16, 3)
    for t in range(16):
        ego_history_xyz[0, 0, t, 0] = (t - 15) * 0.5
    ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 16, 3, 3).clone()
    return {
        "image_frames": image_frames,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": torch.zeros(1, 1, 64, 3),
        "ego_future_rot": torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 64, 3, 3).clone(),
    }


def run_single_experiment(vram_limit_gb: int) -> dict:
    """단일 VRAM 제한 실험 실행"""
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    dummy_size_gb = max(0, total_vram - vram_limit_gb - 0.3)  # 0.3GB CUDA overhead 여유

    print(f"\n{'='*70}")
    print(f"VRAM Limit: {vram_limit_gb} GB (Total: {total_vram:.1f} GB, Dummy: {dummy_size_gb:.1f} GB)")
    print(f"{'='*70}")

    # GPU 초기화
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # Dummy 텐서로 VRAM 점유
    dummy_tensor = None
    if dummy_size_gb > 0.1:
        dummy_elements = int(dummy_size_gb * 1024**3 / 2)  # FP16 = 2 bytes
        try:
            dummy_tensor = torch.empty(dummy_elements, dtype=torch.float16, device="cuda")
            print(f"  Dummy tensor allocated: {dummy_tensor.nelement() * 2 / 1024**3:.2f} GB")
        except RuntimeError as e:
            print(f"  [WARNING] Could not allocate full dummy: {e}")
            dummy_elements = int(dummy_elements * 0.8)
            dummy_tensor = torch.empty(dummy_elements, dtype=torch.float16, device="cuda")
            print(f"  Reduced dummy: {dummy_tensor.nelement() * 2 / 1024**3:.2f} GB")

    available_after_dummy = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
    print(f"  Available VRAM after dummy: {available_after_dummy:.2f} GB")

    result = {
        "vram_limit_gb": vram_limit_gb,
        "dummy_size_gb": round(dummy_size_gb, 2),
        "available_vram_gb": round(available_after_dummy, 2),
    }

    # 모델 로드
    t_load_start = time.time()
    try:
        model = AlpamayoR1.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        model = model.to("cuda")
        t_load_end = time.time()
        result["model_load_time"] = round(t_load_end - t_load_start, 2)
        print(f"  Model loaded: {result['model_load_time']}s | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    except Exception as e:
        print(f"  [FAILED] Model load failed: {e}")
        result["model_load_time"] = None
        result["inference_time"] = None
        result["peak_vram_gb"] = None
        result["error"] = str(e)
        if dummy_tensor is not None:
            del dummy_tensor
        torch.cuda.empty_cache()
        gc.collect()
        return result

    # 입력 준비
    data = create_dummy_data()
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
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

    # 추론
    print("  Running inference...")
    torch.cuda.reset_peak_memory_stats()
    t_infer_start = time.time()
    try:
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
        t_infer_end = time.time()
        result["inference_time"] = round(t_infer_end - t_infer_start, 2)
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["cot_preview"] = str(extra["cot"][0][:200]) if extra.get("cot") else ""
        result["error"] = None
        print(f"  Inference: {result['inference_time']}s | Peak VRAM: {result['peak_vram_gb']} GB")
    except Exception as e:
        t_infer_end = time.time()
        result["inference_time"] = round(t_infer_end - t_infer_start, 2)
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["error"] = str(e)
        print(f"  [FAILED] Inference failed after {result['inference_time']}s: {e}")

    # 정리
    del model
    del model_inputs
    if dummy_tensor is not None:
        del dummy_tensor
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)  # GPU 안정화 대기

    return result


def main():
    print("="*70)
    print("4-bit Alpamayo VRAM Limit Comparison Experiment")
    print("="*70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Test conditions: {VRAM_LIMITS}")

    all_results = []

    for limit in VRAM_LIMITS:
        result = run_single_experiment(limit)
        all_results.append(result)

        # 중간 결과 저장
        with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 최종 결과 출력
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'VRAM Limit':>12} {'Load Time':>12} {'Infer Time':>12} {'Peak VRAM':>12} {'Status':>10}")
    print(f"{'─'*70}")
    for r in all_results:
        load = f"{r['model_load_time']}s" if r['model_load_time'] else "FAIL"
        infer = f"{r['inference_time']}s" if r['inference_time'] else "FAIL"
        peak = f"{r['peak_vram_gb']} GB" if r['peak_vram_gb'] else "N/A"
        status = "OK" if not r.get('error') else "FAIL"
        print(f"{r['vram_limit_gb']:>10} GB {load:>12} {infer:>12} {peak:>12} {status:>10}")

    # JSON 저장
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
