import sys
import os

# experiments/ 디렉토리에서 실행 시 alpamayo 소스를 import할 수 있도록 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alpamayo", "src"))

import torch
import numpy as np
import time
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

print("=" * 60)
print("Alpamayo-R1-10B Dummy Inference Test")
print("=" * 60)

# 1. GPU 상태 확인
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"VRAM Used (before model load): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 2. 더미 데이터 생성
print("\n[1/4] Creating dummy data...")
num_cameras = 4
num_frames = 4
H, W = 320, 576  # Alpamayo 기본 해상도

image_frames = torch.randint(0, 256, (num_cameras, num_frames, 3, H, W), dtype=torch.uint8)

ego_history_xyz = torch.zeros(1, 1, 16, 3)
for t in range(16):
    ego_history_xyz[0, 0, t, 0] = (t - 15) * 0.5

ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 16, 3, 3).clone()

ego_future_xyz = torch.zeros(1, 1, 64, 3)
for t in range(64):
    ego_future_xyz[0, 0, t, 0] = (t + 1) * 0.5

ego_future_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 64, 3, 3).clone()

data = {
    "image_frames": image_frames,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
    "ego_future_xyz": ego_future_xyz,
    "ego_future_rot": ego_future_rot,
}

print("  Done. Dummy data created.")

# 3. 모델 로드
print("\n[2/4] Loading model (nvidia/Alpamayo-R1-10B)...")
load_start = time.time()
try:
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  VRAM Used (after model load): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  VRAM Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
except torch.cuda.OutOfMemoryError as e:
    print(f"  OOM ERROR during model load: {e}")
    print(f"  VRAM Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  VRAM Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    exit(1)

# 4. 입력 준비
print("\n[3/4] Preparing inputs...")
messages = helper.create_message(data["image_frames"].flatten(0, 1))
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")
print(f"  VRAM Used (after input prep): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 5. 추론
print("\n[4/4] Running inference...")
infer_start = time.time()
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
    infer_time = time.time() - infer_start
    print(f"  Inference completed in {infer_time:.1f}s")
    print(f"  VRAM Used (after inference): {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  VRAM Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Chain-of-Causation:\n{extra['cot'][0][:500]}...")
    print(f"\nPredicted trajectory shape: {pred_xyz.shape}")
    print(f"Model load time: {load_time:.1f}s")
    print(f"Inference time: {infer_time:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

except torch.cuda.OutOfMemoryError as e:
    infer_time = time.time() - infer_start
    print(f"  OOM ERROR during inference: {e}")
    print(f"  Time before OOM: {infer_time:.1f}s")
    print(f"  VRAM Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  VRAM Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
except Exception as e:
    infer_time = time.time() - infer_start
    print(f"  ERROR during inference: {type(e).__name__}: {e}")
    print(f"  Time before error: {infer_time:.1f}s")
    print(f"  VRAM Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  VRAM Peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
