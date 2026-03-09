import torch
import traceback
import time

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

print("Loading model with device_map=auto...")
t_start = time.time()
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16, device_map="auto")
t_load = time.time() - t_start
print(f"Model loaded in {t_load:.2f}s")
print(f"Device map: {model.hf_device_map}")

# VRAM 확인
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

processor = helper.get_processor(model.tokenizer)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

first_device = next(model.parameters()).device
print(f"First param device: {first_device}")

model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, first_device)

try:
    torch.cuda.manual_seed_all(42)
    print("Starting inference...")
    t_infer = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    t_infer = time.time() - t_infer
    print(f"Inference SUCCESS in {t_infer:.2f}s")
    print(f"CoT: {extra['cot'][0]}")
except Exception as e:
    print(f"\n=== INFERENCE FAILED ===")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()
