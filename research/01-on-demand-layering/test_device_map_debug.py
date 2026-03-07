#!/usr/bin/env python3
"""Debug script for device_map="auto" masked_scatter assertion error.

Monkey-patches replace_pad_token to print debug info before the actual call,
then runs inference with CUDA_LAUNCH_BLOCKING=1 to get a precise traceback.
"""

import sys
import os
import time
import gc
import traceback

sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")

import torch
import numpy as np


def main():
    print("=" * 80)
    print("Debug: device_map='auto' masked_scatter assertion error")
    print("=" * 80)

    # Clear GPU
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 1: Load model ──
    print("\n[1/3] Loading model with device_map='auto'...")
    t0 = time.time()

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"    Model loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Monkey-patch replace_pad_token ──
    print("\n[2/3] Monkey-patching replace_pad_token for debug output...")

    import alpamayo_r1.models.base_model as base_model_module
    _original_replace_pad_token = base_model_module.replace_pad_token

    def debug_replace_pad_token(input_ids, new_ids, pad_idx):
        """Debug wrapper around replace_pad_token."""
        print("\n" + "=" * 60)
        print("[DEBUG] replace_pad_token called")
        print(f"  input_ids.shape  = {input_ids.shape}")
        print(f"  input_ids.device = {input_ids.device}")
        print(f"  input_ids.dtype  = {input_ids.dtype}")
        print(f"  new_ids.shape    = {new_ids.shape}")
        print(f"  new_ids.device   = {new_ids.device}")
        print(f"  new_ids.dtype    = {new_ids.dtype}")
        print(f"  pad_idx          = {pad_idx}")

        # Move new_ids to input_ids device (as the patched code does)
        new_ids_on_device = new_ids.to(input_ids.device)

        mask = input_ids == pad_idx
        mask_sum = mask.sum().item()
        new_ids_numel = new_ids_on_device.numel()

        print(f"  mask.shape       = {mask.shape}")
        print(f"  mask.sum()       = {mask_sum}  (number of True values)")
        print(f"  new_ids.numel()  = {new_ids_numel}  (total elements in new_ids)")
        print(f"  MATCH?           = {mask_sum == new_ids_numel}  (mask_sum should == new_ids.numel())")

        if mask_sum > new_ids_numel:
            print(f"  *** ERROR: mask has {mask_sum} True values but new_ids only has {new_ids_numel} elements!")
            print(f"  *** This WILL cause 'masked_scatter_size_check: Assertion totalElements <= srcSize failed'")
        elif mask_sum < new_ids_numel:
            print(f"  *** WARNING: mask has fewer True values ({mask_sum}) than new_ids elements ({new_ids_numel})")
            print(f"  *** Extra elements in new_ids will be ignored (this is OK)")

        # Show where pad tokens are located
        if mask.dim() == 2:
            for b in range(mask.shape[0]):
                pad_positions = torch.where(mask[b])[0]
                print(f"  Batch {b}: {pad_positions.numel()} pad tokens at positions: {pad_positions.tolist()[:20]}{'...' if pad_positions.numel() > 20 else ''}")

        # Show some values
        print(f"  input_ids unique values (sample): {torch.unique(input_ids).tolist()[:30]}...")
        print(f"  new_ids values (sample): {new_ids_on_device.flatten().tolist()[:30]}...")
        print("=" * 60)

        # Call original
        return _original_replace_pad_token(input_ids, new_ids_on_device, pad_idx)

    base_model_module.replace_pad_token = debug_replace_pad_token

    # ── Step 3: Run inference ──
    print("\n[3/3] Running dummy inference...")

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

    # Print tokenized data info
    print("\n[DEBUG] Tokenized input info:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: type={type(v)}")

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

    print("\n[DEBUG] Model inputs after to_device('cuda'):")
    for k, v in model_inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    print(f"  {k}.{k2}: shape={v2.shape}, dtype={v2.dtype}, device={v2.device}")

    print("\n[DEBUG] Starting inference call...")
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
        print("\n[SUCCESS] Inference completed!")
        print(f"  pred_xyz.shape = {pred_xyz.shape}")
        print(f"  pred_rot.shape = {pred_rot.shape}")
    except Exception as e:
        print("\n" + "!" * 80)
        print("[FAILED] Inference raised an exception:")
        print("!" * 80)
        traceback.print_exc()
        print("!" * 80)

    # Cleanup
    del model, model_inputs
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
