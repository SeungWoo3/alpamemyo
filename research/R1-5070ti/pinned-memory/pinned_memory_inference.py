"""
Pinned Memory 실험 — On-Demand Layering with Pinned Memory

전략:
1. 모델을 CPU에 FP16으로 로드
2. 전체 모델을 GPU로 이동 (16GB 가능 — 21GB 모델의 일부만 GPU에 맞음)
   대신: 모든 non-VLM 컴포넌트 + 상위 VLM 레이어 → GPU
         하위 NUM_OFFLOAD VLM 레이어 → CPU → pinned memory
3. 간단한 접근: language_model 전체를 GPU로 이동 후 하위 레이어만 CPU로 되돌림
4. 훅 기반 On-Demand H2D (pinned→GPU) + Free (D2H 없음)
"""
import sys
import os
import time
import json
import gc
import threading
import subprocess

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, "/home/avees/workspace/alpamayo/src")

import torch
import torch.nn as nn

RESULTS_DIR = "/home/avees/workspace/research/R1-5070ti/pinned-memory"
os.makedirs(RESULTS_DIR, exist_ok=True)
MODEL_ID = "nvidia/Alpamayo-R1-10B"
NUM_OFFLOAD = 21  # CPU pinned memory로 오프로드할 VLM 레이어 수


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


class LayerOffloadHookPinned:
    """
    Pinned Memory 기반 레이어 오프로드 훅.
    pre_forward: pinned CPU → GPU (H2D, non_blocking=True)
    post_forward: GPU 텐서를 pinned CPU 원본으로 교체 (D2H 없음)
    """

    def __init__(self):
        self.cpu_params = {}
        self.transfer_times = []
        self.hooks = []
        self.pin_time = 0.0

    def pin_cpu_layers(self, layers: nn.ModuleList, layer_indices: list):
        """CPU 레이어 파라미터를 pinned memory로 변환하여 저장"""
        t0 = time.perf_counter()
        total_params = 0
        total_bytes = 0

        for i in layer_indices:
            layer = layers[i]
            first_param = next(layer.parameters(), None)
            if first_param is None or first_param.device.type != "cpu":
                print(f"    [SKIP] Layer {i}: device={first_param.device if first_param else 'none'}")
                continue

            self.cpu_params[i] = {}
            for name, param in layer.named_parameters():
                if param.device.type == "cpu":
                    data = param.data
                    if not data.is_contiguous():
                        data = data.contiguous()
                    if not data.is_pinned():
                        pinned = data.pin_memory()
                    else:
                        pinned = data
                    self.cpu_params[i][name] = pinned
                    param.data = pinned
                    total_params += 1
                    total_bytes += pinned.nelement() * pinned.element_size()

            for name, buf in layer.named_buffers():
                if buf.device.type == "cpu" and not buf.is_pinned():
                    buf.data = buf.data.contiguous().pin_memory()

        t1 = time.perf_counter()
        self.pin_time = t1 - t0
        total_gb = total_bytes / 1024**3
        pinned_count = len(self.cpu_params)
        print(f"    Pinned {pinned_count} layers ({total_params} params, {total_gb:.2f} GB)")
        print(f"    Pinning time: {self.pin_time:.2f}s")
        return pinned_count

    def _pre_forward(self, module, input, layer_idx):
        """H2D: Pinned CPU → GPU"""
        if layer_idx not in self.cpu_params:
            return
        t0 = time.perf_counter()

        param_dict = dict(module.named_parameters())
        for name, pinned_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict:
                param_dict[name].data = pinned_tensor.to("cuda", non_blocking=True)

        for name, buf in module.named_buffers(recurse=True):
            if buf.device.type == "cpu":
                buf.data = buf.data.to("cuda", non_blocking=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.transfer_times.append(("H2D", t1 - t0))

    def _post_forward(self, module, input, output, layer_idx):
        """Free: GPU 텐서를 pinned CPU 원본으로 교체"""
        if layer_idx not in self.cpu_params:
            return
        t0 = time.perf_counter()

        param_dict = dict(module.named_parameters())
        for name, pinned_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict and param_dict[name].device.type == "cuda":
                param_dict[name].data = pinned_tensor

        for name, buf in module.named_buffers(recurse=True):
            if buf.device.type == "cuda":
                buf.data = buf.data.cpu()

        torch.cuda.empty_cache()
        t1 = time.perf_counter()
        self.transfer_times.append(("FREE", t1 - t0))

    def register_hooks(self, layers: nn.ModuleList, layer_indices: list):
        for i in layer_indices:
            if i not in self.cpu_params:
                continue
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
            "pin_time_s": round(self.pin_time, 3),
        }


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,pcie.link.gen.current,pcie.link.width.current,power.draw"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def get_ram_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_experiment():
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    config_name = "pinned_memory_demand_layering"
    print(f"\n{'='*70}")
    print(f"Experiment: {config_name}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram_gb:.1f} GB")
    print(f"GPU Info: {get_gpu_info()}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")

    result = {"config": config_name, "gpu": torch.cuda.get_device_name(0)}
    monitor = VRAMMonitor(interval=0.1)
    monitor.start()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Phase 1: 데이터 로드
    print("\n[Phase 1] Loading dataset...")
    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    print(f"  Dataset loaded")

    # Phase 2: 모델 CPU 로드
    print("\n[Phase 2] Loading model to CPU (FP16)...")
    model_load_start = time.time() - monitor.t0
    t_model = time.time()
    model = AlpamayoR1.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    t_load = time.time() - t_model
    result["cuda_load_time"] = round(t_load, 2)
    result["model_load_start_t"] = model_load_start
    print(f"  CPU load time: {t_load:.2f}s | RAM: {get_ram_gb():.1f} GB")

    # Phase 3: 선택적 GPU 이동
    print(f"\n[Phase 3] Moving model to GPU (with {NUM_OFFLOAD} VLM layer CPU offload)...")
    t3 = time.time()
    hook = LayerOffloadHookPinned()

    try:
        vlm_layers = model.vlm.model.language_model.layers
        total_layers = len(vlm_layers)
        print(f"  Total VLM layers: {total_layers}")

        # 전략: 모든 컴포넌트를 GPU로 이동 (순서 중요)
        # 1) Vision encoder
        print("  Step 1: vision encoder → GPU")
        model.vlm.model.visual.to("cuda")
        print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # 2) Language model의 embedding + norm + rotary (레이어 제외)
        print("  Step 2: language_model non-layer components → GPU")
        lm = model.vlm.model.language_model
        lm.embed_tokens.to("cuda")
        lm.norm.to("cuda")
        if hasattr(lm, 'rotary_emb'):
            lm.rotary_emb.to("cuda")
        print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # 3) LM head
        print("  Step 3: lm_head → GPU")
        model.vlm.lm_head.to("cuda")
        print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # 4) Expert decoder, diffusion
        print("  Step 4: expert_decoder + diffusion → GPU")
        if hasattr(model, 'expert'):
            model.expert.to("cuda")
        if hasattr(model, 'diffusion'):
            model.diffusion.to("cuda")
        if hasattr(model, 'action_in_proj'):
            model.action_in_proj.to("cuda")
        if hasattr(model, 'action_out_proj'):
            model.action_out_proj.to("cuda")
        print(f"    VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # 5) 상위 VLM 레이어 (NUM_OFFLOAD ~ 35) → GPU
        print(f"  Step 5: VLM layers {NUM_OFFLOAD}-{total_layers-1} → GPU")
        for i in range(NUM_OFFLOAD, total_layers):
            vlm_layers[i].to("cuda")
            if (i - NUM_OFFLOAD + 1) % 3 == 0:
                torch.cuda.empty_cache()
                v = torch.cuda.memory_allocated() / 1024**3
                print(f"    Layer {i} → GPU | VRAM: {v:.2f} GB")

        torch.cuda.empty_cache()
        gc.collect()
        gpu_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU after all moves: {gpu_gb:.2f} GB")
        result["gpu_resident_gb"] = round(gpu_gb, 2)

        # 6) 하위 VLM 레이어 (0 ~ NUM_OFFLOAD-1) → Pinned Memory
        cpu_layer_indices = list(range(NUM_OFFLOAD))
        print(f"  Step 6: Pinning VLM layers 0-{NUM_OFFLOAD-1} (CPU)...")
        pinned_count = hook.pin_cpu_layers(vlm_layers, cpu_layer_indices)

        result["setup_time"] = round(time.time() - t3, 2)
        result["vlm_layers_offloaded"] = pinned_count
        result["vlm_layers_total"] = total_layers
        result["model_load_end_t"] = time.time() - monitor.t0

        hook.register_hooks(vlm_layers, cpu_layer_indices)
        print(f"  Hooks registered for {pinned_count} layers")
        print(f"  Setup total: {result['setup_time']}s")

    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback; traceback.print_exc()
        monitor.stop()
        del model; torch.cuda.empty_cache(); gc.collect()
        return {"error": str(e)}

    # Phase 4: 입력 준비
    print("\n[Phase 4] Preparing inputs...")
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

    # Phase 5: 추론
    print("\n[Phase 5] Running inference (pinned H2D + free)...")
    print(f"  GPU Info: {get_gpu_info()}")
    torch.cuda.reset_peak_memory_stats()
    inference_start = time.time() - monitor.t0
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
        result["cot"] = str(extra["cot"][0])
        result["error"] = None

        import numpy as np
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        result["minADE"] = round(float(diff.min()), 4)

        print(f"  Inference: {result['inference_time']}s")
        print(f"  Peak VRAM: {result['peak_vram_gb']} GB")
        print(f"  minADE: {result['minADE']}m")
    except Exception as e:
        t5 = time.time()
        result["inference_time"] = round(t5 - t4, 2)
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        result["error"] = str(e)
        print(f"  [FAILED] {e}")
        import traceback; traceback.print_exc()

    result["inference_start_t"] = inference_start
    result["inference_end_t"] = time.time() - monitor.t0
    result["gpu_info_after"] = get_gpu_info()
    result["transfer_stats"] = hook.get_stats()
    result["total_time"] = round(time.time() - t0, 2)

    monitor.stop()
    result["peak_vram_allocated"] = max(monitor.allocated) if monitor.allocated else 0
    result["peak_vram_reserved"] = max(monitor.reserved) if monitor.reserved else 0

    vram_data = monitor.to_dict()
    hook.remove_hooks()
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 결과 출력
    print(f"\n{'='*70}")
    print("RESULTS (Pinned Memory)")
    print(f"{'='*70}")
    stats = result.get("transfer_stats", {})
    print(f"  Load: {result['cuda_load_time']}s | Setup: {result.get('setup_time','N/A')}s")
    print(f"  Inference: {result.get('inference_time', 'N/A')}s")
    print(f"  Peak VRAM: {result.get('peak_vram_gb', 'N/A')} GB")
    if stats:
        print(f"  H2D: {stats.get('h2d_total_s')}s ({stats.get('h2d_avg_ms')}ms avg)")
        print(f"  FREE: {stats.get('free_total_s')}s ({stats.get('free_avg_ms')}ms avg)")
        print(f"  Pin time: {stats.get('pin_time_s')}s")
    if result.get("error") is None:
        print(f"  minADE: {result.get('minADE', 'N/A')}m")

    # 저장
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    vram_path = os.path.join(RESULTS_DIR, "vram_timeline.json")
    with open(vram_path, "w") as f:
        json.dump(vram_data, f)

    print(f"\nResults saved: {results_path}")
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("Alpamayo-R1-10B — Pinned Memory On-Demand Layering")
    print("=" * 70)
    run_experiment()
