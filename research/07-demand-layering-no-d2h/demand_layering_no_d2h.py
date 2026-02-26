"""
Demand Layering without D2H — GPU Free Only

기존 demand_layering.py에서 D2H(module.to("cpu"))를 제거하고,
CPU에 보관된 원본 파라미터를 복원 + GPU 메모리 해제만 수행.

변경점:
- 오프로드 전에 CPU 파라미터 원본을 별도 보관
- pre_forward: CPU 원본 → GPU 복사 (H2D)
- post_forward: GPU 파라미터를 CPU 원본으로 교체 (free만, D2H 없음)
"""
import sys
import os
import time
import json
import gc

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


class LayerOffloadHookNoD2H:
    """
    D2H 없는 레이어 오프로드 훅.

    CPU에 파라미터 원본을 보관하고:
    - pre_forward: CPU 원본 → GPU 복사 (H2D)
    - post_forward: param.data를 CPU 원본으로 교체 (GPU 텐서 참조 해제 → free)
    """

    def __init__(self):
        self.cpu_params = {}   # layer_idx -> {name: cpu_tensor}
        self.transfer_times = []
        self.hooks = []

    def save_cpu_refs(self, layers: nn.ModuleList, num_offload: int):
        """오프로드 후 CPU 파라미터 참조를 저장 (clone 없음 → 추가 메모리 0)"""
        for i in range(num_offload):
            layer = layers[i]
            self.cpu_params[i] = {}
            for name, param in layer.named_parameters():
                # CPU 텐서의 참조만 저장 (clone 아님 → 메모리 추가 없음)
                self.cpu_params[i][name] = param.data
        total_params = sum(len(v) for v in self.cpu_params.values())
        print(f"    Saved CPU refs for {num_offload} layers ({total_params} total params, no extra memory)")

    def _pre_forward(self, module, input, layer_idx):
        """H2D: CPU 원본에서 GPU로 복사"""
        t0 = time.perf_counter()

        param_dict = dict(module.named_parameters())
        for name, cpu_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict:
                param_dict[name].data = cpu_tensor.to("cuda", non_blocking=False)

        # 버퍼도 이동 (layer_norm 등)
        for name, buf in module.named_buffers():
            if buf.device.type == "cpu":
                buf.data = buf.data.to("cuda", non_blocking=False)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.transfer_times.append(("H2D", t1 - t0))

    def _post_forward(self, module, input, output, layer_idx):
        """GPU Free: CPU 원본 복원 (D2H copy 없음)"""
        t0 = time.perf_counter()

        # GPU 텐서 참조를 CPU 원본으로 교체 → GPU 메모리 자동 해제
        param_dict = dict(module.named_parameters())
        for name, cpu_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict:
                param_dict[name].data = cpu_tensor

        # 버퍼도 CPU로
        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                buf.data = buf.data.cpu()

        torch.cuda.empty_cache()
        t1 = time.perf_counter()
        self.transfer_times.append(("FREE", t1 - t0))

    def register_hooks(self, layers: nn.ModuleList, num_offload: int):
        for i in range(num_offload):
            layer = layers[i]
            pre = layer.register_forward_pre_hook(
                lambda mod, inp, idx=i: self._pre_forward(mod, inp, idx)
            )
            post = layer.register_forward_hook(
                lambda mod, inp, out, idx=i: self._post_forward(mod, inp, out, idx)
            )
            self.hooks.append(pre)
            self.hooks.append(post)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_stats(self):
        h2d = [t for d, t in self.transfer_times if d == "H2D"]
        free = [t for d, t in self.transfer_times if d == "FREE"]
        return {
            "total_transfers": len(self.transfer_times),
            "h2d_count": len(h2d),
            "free_count": len(free),
            "h2d_total_s": round(sum(h2d), 3),
            "free_total_s": round(sum(free), 3),
            "h2d_avg_ms": round(sum(h2d) / max(len(h2d), 1) * 1000, 2),
            "free_avg_ms": round(sum(free) / max(len(free), 1) * 1000, 2),
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
    """CUDA에 직접 모델 생성 + safetensors에서 가중치 로드"""
    from alpamayo_r1.config import AlpamayoR1Config

    print("  Loading config...")
    config = AlpamayoR1Config.from_pretrained(MODEL_ID)

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
            except (AttributeError, IndexError, KeyError):
                pass

        del shard
        torch.cuda.empty_cache()
        print(f"    {shard_name}: loaded | GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"  Loaded {loaded_count}/{len(weight_map)} weights")
    return model


def get_ram_usage():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def offload_layers_to_cpu(model, hook: LayerOffloadHookNoD2H, num_offload=30):
    """VLM 레이어를 CPU로 이동 + CPU 원본 저장"""
    vlm_layers = model.vlm.model.language_model.layers
    total_layers = len(vlm_layers)
    num_keep = total_layers - num_offload

    print(f"  Saving CPU copies for {num_offload} layers...")

    # 먼저 CPU로 이동
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

    # CPU 참조 저장 (clone 없음)
    hook.save_cpu_refs(vlm_layers, num_offload)

    gpu_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Final GPU: {gpu_gb:.2f} GB")
    print(f"  VLM layers 0-{num_offload-1}: CPU (on-demand, no D2H)")
    print(f"  VLM layers {num_offload}-{total_layers-1}: GPU (resident)")
    print(f"  Expert: GPU (resident)")
    return gpu_gb, num_offload


def run_experiment():
    """D2H 제거 Demand Layering 실험"""
    config_name = "demand_layering_no_d2h"

    print(f"\n{'='*70}")
    print(f"Experiment: {config_name}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram_gb:.1f} GB")

    result = {"config": config_name}

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Phase 1: 모델 로드
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

    # Phase 2: 레이어 오프로드 + CPU 원본 저장
    print("\n[Phase 2] Setting up demand layering (No D2H)...")
    t2 = time.time()

    hook = LayerOffloadHookNoD2H()

    try:
        gpu_resident, num_offloaded = offload_layers_to_cpu(model, hook, num_offload=30)
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

    # Hook 등록
    vlm_layers = model.vlm.model.language_model.layers
    hook.register_hooks(vlm_layers, num_offloaded)
    num_kept = len(vlm_layers) - num_offloaded
    print(f"  VLM hooks: {num_offloaded} layers (H2D + free, no D2H)")
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
    print("\n[Phase 4] Running inference (H2D + free, no D2H)...")
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
    result["vlm_transfer_stats"] = hook.get_stats()
    result["expert_transfer_stats"] = {"note": "Expert on GPU (no offloading)"}
    result["total_time"] = round(t5 - t0, 2)

    hook.remove_hooks()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 출력
    print(f"\n{'='*70}")
    print("RESULTS (No D2H)")
    print(f"{'='*70}")
    print(f"Config: {config_name}")
    print(f"Total time: {result['total_time']}s")
    print(f"  Load: {result['cuda_load_time']}s | Setup: {result.get('setup_time','N/A')}s")
    print(f"  Inference: {result.get('inference_time', 'N/A')}s")
    print(f"Peak VRAM: {result.get('peak_vram_gb', 'N/A')} GB")

    stats = result.get("vlm_transfer_stats", {})
    if stats and "h2d_count" in stats:
        print(f"\nVLM transfers: {stats['h2d_count']} H2D, {stats['free_count']} FREE (no D2H)")
        print(f"  H2D:  {stats['h2d_total_s']}s total (avg {stats['h2d_avg_ms']}ms)")
        print(f"  FREE: {stats['free_total_s']}s total (avg {stats['free_avg_ms']}ms)")

    # 비교
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"{'Method':<40} {'Infer Time':>12} {'Peak VRAM':>12}")
    print(f"{'─'*65}")
    print(f"{'FP16 Unified Memory (baseline)':<40} {'273.79s':>12} {'21.52 GB':>12}")
    print(f"{'Demand Layering (with D2H)':<40} {'116.51s':>12} {'11.03 GB':>12}")
    t = f"{result.get('inference_time', 'N/A')}s" if result.get('inference_time') else "FAIL"
    v = f"{result.get('peak_vram_gb', 'N/A')} GB" if result.get('peak_vram_gb') else "N/A"
    print(f"{'Demand Layering (no D2H, FREE only)':<40} {t:>12} {v:>12}")
    print(f"{'이론 예측 (D2H 제거)':<40} {'~46s':>12}")

    return result


def main():
    print("=" * 70)
    print("Alpamayo-R1-10B Demand Layering — No D2H (Free Only)")
    print("=" * 70)
    print(f"RAM used: {get_ram_usage():.1f} GB")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    result = run_experiment()

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump([result], f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
