"""
Demand Layering for Alpamayo-R1-10B

레이어별 On-Demand 로딩으로 12GB VRAM 내에서 BF16 Alpamayo 추론.

핵심 전략:
- System RAM 16GB로 FP16 모델(20GB)을 CPU에 올릴 수 없음
- torch.set_default_device("cuda")로 모델을 직접 CUDA에 생성 (Unified Memory)
- safetensors에서 shard별로 가중치를 CUDA로 직접 로드
- VLM/Expert 레이어를 CPU로 이동 후 forward hook으로 on-demand 로드
"""
import sys
import os
import time
import json
import gc
import glob as glob_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "alpamayo", "src"))

import torch
import torch.nn as nn
from safetensors.torch import load_file

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "nvidia/Alpamayo-R1-10B"
CHECKPOINT_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/"
    "snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
)


class LayerOffloadHook:
    """Forward hook으로 레이어를 on-demand GPU에 로드/언로드"""

    def __init__(self):
        self.transfer_times = []
        self.hooks = []

    def _move_to_gpu(self, module):
        t0 = time.perf_counter()
        module.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.transfer_times.append(("H2D", t1 - t0))

    def _move_to_cpu(self, module):
        t0 = time.perf_counter()
        module.to("cpu", non_blocking=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.transfer_times.append(("D2H", t1 - t0))

    def register_hooks(self, layers: nn.ModuleList):
        for i, layer in enumerate(layers):
            pre = layer.register_forward_pre_hook(
                lambda mod, inp, idx=i: self._pre_forward(mod, inp, idx)
            )
            post = layer.register_forward_hook(
                lambda mod, inp, out, idx=i: self._post_forward(mod, inp, out, idx)
            )
            self.hooks.append(pre)
            self.hooks.append(post)

    def _pre_forward(self, module, input, layer_idx):
        self._move_to_gpu(module)

    def _post_forward(self, module, input, output, layer_idx):
        self._move_to_cpu(module)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_stats(self):
        h2d = [t for d, t in self.transfer_times if d == "H2D"]
        d2h = [t for d, t in self.transfer_times if d == "D2H"]
        return {
            "total_transfers": len(self.transfer_times),
            "h2d_count": len(h2d),
            "d2h_count": len(d2h),
            "h2d_total_s": round(sum(h2d), 3),
            "d2h_total_s": round(sum(d2h), 3),
            "h2d_avg_ms": round(sum(h2d) / max(len(h2d), 1) * 1000, 2),
            "d2h_avg_ms": round(sum(d2h) / max(len(d2h), 1) * 1000, 2),
        }


def create_dummy_data():
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


def load_model_direct_cuda():
    """
    CUDA에 직접 모델 생성 + safetensors에서 가중치 로드.
    CPU RAM 사용 최소화.
    """
    from alpamayo_r1.config import AlpamayoR1Config

    # 1. Config 로드 (CPU only, 작은 JSON)
    print("  Loading config...")
    config = AlpamayoR1Config.from_pretrained(MODEL_ID)

    # 2. 모델을 CUDA에 직접 생성 (Unified Memory)
    print("  Creating model on CUDA (Unified Memory)...")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    model = AlpamayoR1(config)
    model.eval()

    torch.set_default_device(None)
    torch.set_default_dtype(torch.float32)

    gpu_after_init = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model created on CUDA: {gpu_after_init:.2f} GB")

    # 3. safetensors에서 가중치 로드 (shard별로)
    print("  Loading weights from safetensors...")
    index_path = os.path.join(CHECKPOINT_PATH, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    weight_map = index["weight_map"]

    loaded_count = 0
    for shard_name in shard_files:
        shard_path = os.path.join(CHECKPOINT_PATH, shard_name)
        shard = load_file(shard_path, device="cuda")

        for key, tensor in shard.items():
            # Navigate model hierarchy to find the parameter
            parts = key.split(".")
            obj = model
            try:
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                param_name = parts[-1]

                target = getattr(obj, param_name, None)
                if target is not None:
                    if isinstance(target, nn.Parameter):
                        target.data.copy_(tensor)
                    else:
                        setattr(obj, param_name, tensor)
                    loaded_count += 1
            except (AttributeError, IndexError, KeyError) as e:
                pass  # Some keys may not match directly

        del shard
        torch.cuda.empty_cache()
        print(f"    {shard_name}: loaded | GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"  Loaded {loaded_count}/{len(weight_map)} weights")
    return model


def offload_layers_to_cpu(model, num_offload=30):
    """
    VLM 레이어를 부분적으로 CPU로 이동. Expert는 GPU에 유지.

    전략:
    - CPU RAM 제약(16GB)으로 36개 전부 오프로드 시 추론 OOM 발생
    - num_offload개만 CPU로, 나머지는 GPU에 유지
    - Expert(4.56GB)는 항상 GPU에 유지

    메모리 계산 (num_offload=30, keep=6):
    - GPU: 7.71 + 6×0.42 = 10.23GB + active_layer(0.42) + dynamic(0.87) ≈ 11.52GB (<12GB)
    - CPU: 30×0.42 = 12.6GB → RAM ~12.6GB → free ~2.9GB (inference 충분)
    """
    vlm_layers = model.vlm.model.language_model.layers
    total_layers = len(vlm_layers)
    num_keep = total_layers - num_offload

    print(f"  Partial offload: {num_offload}/{total_layers} VLM layers to CPU, "
          f"{num_keep} on GPU | Expert on GPU")

    for i in range(num_offload):
        vlm_layers[i].to("cpu")
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"    VLM {i+1}/{num_offload} offloaded | "
                  f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB | "
                  f"RAM: {get_ram_usage():.1f} GB")

    torch.cuda.empty_cache()
    gc.collect()
    gpu_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Final GPU: {gpu_gb:.2f} GB")
    print(f"  VLM layers 0-{num_offload-1}: CPU (on-demand)")
    print(f"  VLM layers {num_offload}-{total_layers-1}: GPU (resident)")
    print(f"  Expert: GPU (resident)")
    return gpu_gb, num_offload


def get_ram_usage():
    """현재 프로세스의 RSS (GB)"""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # KB to GB


def run_demand_layering_experiment():
    """Demand Layering 실험"""
    config_name = "demand_layering_direct_cuda"

    print(f"\n{'='*70}")
    print(f"Demand Layering Experiment: {config_name}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram_gb:.1f} GB")

    result = {"config": config_name}

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Phase 1: 모델 로드 (직접 CUDA 생성)
    print("\n[Phase 1] Direct CUDA model creation + weight loading...")
    try:
        model = load_model_direct_cuda()
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        result["cuda_load_time"] = round(time.time() - t0, 2)
        return result

    t_load = time.time()
    result["cuda_load_time"] = round(t_load - t0, 2)
    print(f"  Total load time: {result['cuda_load_time']}s")

    # Phase 2: 레이어를 CPU로 이동
    print("\n[Phase 2] Setting up demand layering...")
    t2 = time.time()
    try:
        gpu_resident, num_offloaded = offload_layers_to_cpu(model, num_offload=30)
    except Exception as e:
        print(f"  [FAILED] Offload failed: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        result["setup_time"] = round(time.time() - t2, 2)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return result

    t3 = time.time()
    result["setup_time"] = round(t3 - t2, 2)
    result["gpu_resident_gb"] = round(gpu_resident, 2)
    result["vlm_layers_offloaded"] = num_offloaded
    result["vlm_layers_gpu_resident"] = len(model.vlm.model.language_model.layers) - num_offloaded

    # Hook 등록 (오프로드된 VLM 레이어만)
    vlm_hook = LayerOffloadHook()

    vlm_layers = model.vlm.model.language_model.layers
    # 오프로드된 레이어(0~num_offloaded-1)에만 hook 등록
    offloaded_layers = nn.ModuleList([vlm_layers[i] for i in range(num_offloaded)])
    vlm_hook.register_hooks(offloaded_layers)
    num_kept = len(vlm_layers) - num_offloaded
    print(f"  VLM hooks: {num_offloaded} layers (on-demand GPU load)")
    print(f"  VLM GPU-resident: {num_kept} layers | Expert: {len(model.expert.layers)} layers")

    # Phase 3: 입력 준비
    print("\n[Phase 3] Preparing inputs...")
    from alpamayo_r1 import helper
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

    # Phase 4: 추론
    print("\n[Phase 4] Running inference with demand layering...")
    torch.cuda.reset_peak_memory_stats()
    t4 = time.time()

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
        t5 = time.time()
        result["inference_time"] = round(t5 - t4, 2)
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["cot_preview"] = str(extra["cot"][0][:200]) if extra.get("cot") else ""
        result["error"] = None
        print(f"  Inference: {result['inference_time']}s")
        print(f"  Peak VRAM: {result['peak_vram_gb']} GB")
    except Exception as e:
        t5 = time.time()
        result["inference_time"] = round(t5 - t4, 2)
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["error"] = str(e)
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()

    # Phase 5: 통계
    result["vlm_transfer_stats"] = vlm_hook.get_stats()
    result["expert_transfer_stats"] = {"note": "Expert on GPU (no offloading)"}
    result["total_time"] = round(t5 - t0, 2)

    vlm_hook.remove_hooks()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 출력
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Config: {config_name}")
    print(f"Total time: {result['total_time']}s")
    print(f"  Load: {result['cuda_load_time']}s | Setup: {result.get('setup_time','N/A')}s")
    print(f"  Inference: {result.get('inference_time', 'N/A')}s")
    print(f"Peak VRAM: {result.get('peak_vram_gb', 'N/A')} GB")
    print(f"GPU-resident: {result.get('gpu_resident_gb', 'N/A')} GB")

    vlm_stats = result.get("vlm_transfer_stats", {})
    if vlm_stats and "h2d_count" in vlm_stats:
        print(f"\nVLM transfers: {vlm_stats['h2d_count']} H2D, {vlm_stats['d2h_count']} D2H")
        print(f"  H2D: {vlm_stats['h2d_total_s']}s total (avg {vlm_stats['h2d_avg_ms']}ms)")
        print(f"  D2H: {vlm_stats['d2h_total_s']}s total (avg {vlm_stats['d2h_avg_ms']}ms)")
    print(f"Expert: GPU-resident (no offloading needed)")

    if result.get("error"):
        print(f"\n[ERROR]: {result['error']}")

    return result


def main():
    print("=" * 70)
    print("Alpamayo-R1-10B Demand Layering Implementation")
    print("=" * 70)
    print(f"RAM free: {get_ram_usage():.1f} GB used by this process")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    all_results = []
    result = run_demand_layering_experiment()
    all_results.append(result)

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")

    # Baseline 비교
    print(f"\n{'='*70}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*70}")
    print(f"{'Method':<35} {'Infer Time':>12} {'Peak VRAM':>12}")
    print(f"{'─'*60}")
    print(f"{'FP16 Unified Memory (baseline)':<35} {'273.79s':>12} {'21.52 GB':>12}")
    print(f"{'4-bit (no swap)':<35} {'4.79s':>12} {'8.87 GB':>12}")
    for r in all_results:
        t = f"{r.get('inference_time', 'N/A')}s" if r.get('inference_time') is not None else "FAIL"
        v = f"{r.get('peak_vram_gb', 'N/A')} GB" if r.get('peak_vram_gb') is not None else "N/A"
        print(f"{r['config']:<35} {t:>12} {v:>12}")


if __name__ == "__main__":
    main()
