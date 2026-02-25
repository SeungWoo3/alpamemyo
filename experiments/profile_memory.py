import sys
import os

# experiments/ 디렉토리에서 실행 시 alpamayo 소스를 import할 수 있도록 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alpamayo", "src"))

import torch
import time
import threading
import csv
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# === 메모리 모니터링 스레드 ===
memory_log = []
monitoring = True

def monitor_memory(interval=0.5):
    """0.5초 간격으로 GPU 메모리 상태 기록"""
    while monitoring:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        memory_log.append({
            "time": time.time(),
            "allocated_gb": round(allocated, 3),
            "reserved_gb": round(reserved, 3),
            "phase": current_phase,
        })
        time.sleep(interval)

current_phase = "init"

# === 모니터링 시작 ===
monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
monitor_thread.start()

print("=" * 60)
print("Alpamayo-R1-10B Memory Profiling")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# === Phase 1: 더미 데이터 생성 ===
current_phase = "data_creation"
t0 = time.time()

num_cameras = 4
num_frames = 4
H, W = 320, 576
image_frames = torch.randint(0, 256, (num_cameras, num_frames, 3, H, W), dtype=torch.uint8)
ego_history_xyz = torch.zeros(1, 1, 16, 3)
for t in range(16):
    ego_history_xyz[0, 0, t, 0] = (t - 15) * 0.5
ego_history_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 16, 3, 3).clone()
ego_future_xyz = torch.zeros(1, 1, 64, 3)
ego_future_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, 64, 3, 3).clone()

data = {
    "image_frames": image_frames,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
    "ego_future_xyz": ego_future_xyz,
    "ego_future_rot": ego_future_rot,
}
t1 = time.time()
print(f"\n[Phase 1] Data creation: {t1-t0:.2f}s")

# === Phase 2: 모델 로드 (CPU) ===
current_phase = "model_load_cpu"
t2 = time.time()
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16)
t3 = time.time()
print(f"[Phase 2] Model load (CPU): {t3-t2:.2f}s | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# === Phase 3: 모델 → GPU 이동 ===
current_phase = "model_to_cuda"
t4 = time.time()
model = model.to("cuda")
t5 = time.time()
print(f"[Phase 3] Model .to(cuda): {t5-t4:.2f}s | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# === Phase 4: 프로세서/입력 준비 ===
current_phase = "input_preparation"
t6 = time.time()
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
t7 = time.time()
print(f"[Phase 4] Input preparation: {t7-t6:.2f}s | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# === Phase 5: VLM 추론 (토큰 생성) ===
current_phase = "vlm_inference"
t8 = time.time()
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
t9 = time.time()
print(f"[Phase 5] VLM inference: {t9-t8:.2f}s | VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# === 모니터링 종료 ===
monitoring = False
time.sleep(1)

# === 결과 출력 ===
total = t9 - t0
print(f"\n{'=' * 60}")
print("PROFILING RESULTS")
print(f"{'=' * 60}")
print(f"Phase 1 (Data creation):     {t1-t0:>8.2f}s ({(t1-t0)/total*100:>5.1f}%)")
print(f"Phase 2 (Model load CPU):    {t3-t2:>8.2f}s ({(t3-t2)/total*100:>5.1f}%)")
print(f"Phase 3 (Model .to(cuda)):   {t5-t4:>8.2f}s ({(t5-t4)/total*100:>5.1f}%)")
print(f"Phase 4 (Input preparation): {t7-t6:>8.2f}s ({(t7-t6)/total*100:>5.1f}%)")
print(f"Phase 5 (VLM inference):     {t9-t8:>8.2f}s ({(t9-t8)/total*100:>5.1f}%)")
print(f"{'─' * 60}")
print(f"TOTAL:                       {total:>8.2f}s")
print(f"\nPeak VRAM (allocated): {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print(f"Peak VRAM (reserved):  {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
print(f"\nCoC output: {extra['cot'][0][:200]}...")

# === 메모리 로그 저장 ===
log_path = os.path.join(os.path.dirname(__file__), "memory_profile.csv")
start_time = memory_log[0]["time"] if memory_log else 0
with open(log_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["elapsed_s", "allocated_gb", "reserved_gb", "phase"])
    writer.writeheader()
    for entry in memory_log:
        writer.writerow({
            "elapsed_s": round(entry["time"] - start_time, 2),
            "allocated_gb": entry["allocated_gb"],
            "reserved_gb": entry["reserved_gb"],
            "phase": entry["phase"],
        })
print(f"\nMemory log saved: {log_path} ({len(memory_log)} samples)")
