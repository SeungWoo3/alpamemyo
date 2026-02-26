"""
Async 2-Stream Pipeline — Method 4 실험

방법 1(동기 H2D + Free, 43.38s)의 H2D 전송을 계산과 비동기 오버랩하여 가속.
2개의 CUDA Stream + Double-Buffered Pinned Staging Buffer 사용:
  - compute stream (default): 레이어 forward pass
  - transfer stream: 다음 레이어 H2D prefetch (pinned→GPU async DMA)

Variant A: 기본 async (pageable, non_blocking) — 대조군
Variant C: Staged pinned buffer — 소량(~800MB) pinned buffer로 진정한 async DMA
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


# ══════════════════════════════════════════════════════════════
# Variant A: Basic Async (Pageable) — 대조군
# ══════════════════════════════════════════════════════════════

class AsyncPipelineHookPageable:
    """
    Pageable memory 기반 비동기 시도 (대조군).
    non_blocking=True를 사용하지만, pageable에서는 CUDA가 내부 동기 처리.
    """

    def __init__(self):
        self.cpu_params = {}
        self.vlm_layers = None
        self.num_offload = 0
        self.transfer_stream = torch.cuda.Stream()
        self.prefetch_events = {}
        self.prefetch_active = {}
        self.transfer_times = []
        self.current_pass = -1
        self.hooks = []

    def save_cpu_refs(self, layers, num_offload):
        self.vlm_layers = layers
        self.num_offload = num_offload
        for i in range(num_offload):
            self.cpu_params[i] = {}
            for name, param in layers[i].named_parameters():
                self.cpu_params[i][name] = param.data
        print(f"    Saved CPU refs for {num_offload} layers")

    def _start_prefetch(self, layer_idx):
        layer = self.vlm_layers[layer_idx]
        param_dict = dict(layer.named_parameters())
        event = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            for name, cpu_tensor in self.cpu_params[layer_idx].items():
                if name in param_dict:
                    param_dict[name].data = cpu_tensor.to("cuda", non_blocking=True)
            for name, buf in layer.named_buffers():
                if buf.device.type == "cpu":
                    buf.data = buf.data.to("cuda", non_blocking=True)
            event.record()
        self.prefetch_events[layer_idx] = event
        self.prefetch_active[layer_idx] = True

    def _pre_forward(self, module, input, layer_idx):
        t0 = time.perf_counter()
        if layer_idx == 0:
            self.current_pass += 1
            self.prefetch_active.clear()
            self.prefetch_events.clear()

        if self.prefetch_active.get(layer_idx, False):
            torch.cuda.current_stream().wait_event(self.prefetch_events[layer_idx])
            self.prefetch_active[layer_idx] = False
            t1 = time.perf_counter()
            self.transfer_times.append(("WAIT_H2D", t1 - t0, layer_idx, self.current_pass))
        else:
            param_dict = dict(module.named_parameters())
            for name, cpu_tensor in self.cpu_params[layer_idx].items():
                if name in param_dict:
                    param_dict[name].data = cpu_tensor.to("cuda", non_blocking=False)
            for name, buf in module.named_buffers():
                if buf.device.type == "cpu":
                    buf.data = buf.data.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.transfer_times.append(("SYNC_H2D", t1 - t0, layer_idx, self.current_pass))

    def _post_forward(self, module, input, output, layer_idx):
        next_idx = layer_idx + 1
        if next_idx < self.num_offload:
            t0 = time.perf_counter()
            self._start_prefetch(next_idx)
            t1 = time.perf_counter()
            self.transfer_times.append(("PREFETCH_LAUNCH", t1 - t0, next_idx, self.current_pass))

        t2 = time.perf_counter()
        param_dict = dict(module.named_parameters())
        for name, cpu_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict:
                param_dict[name].data = cpu_tensor
        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                buf.data = buf.data.cpu()
        t3 = time.perf_counter()
        self.transfer_times.append(("FREE", t3 - t2, layer_idx, self.current_pass))

    def register_hooks(self, layers, num_offload):
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
        sync_h2d = [t for ty, t, li, p in self.transfer_times if ty == "SYNC_H2D"]
        wait_h2d = [t for ty, t, li, p in self.transfer_times if ty == "WAIT_H2D"]
        free = [t for ty, t, li, p in self.transfer_times if ty == "FREE"]
        prefetch = [t for ty, t, li, p in self.transfer_times if ty == "PREFETCH_LAUNCH"]
        return {
            "total_passes": self.current_pass + 1,
            "sync_h2d_count": len(sync_h2d),
            "async_h2d_count": len(wait_h2d),
            "free_count": len(free),
            "prefetch_count": len(prefetch),
            "sync_h2d_total_s": round(sum(sync_h2d), 3),
            "async_h2d_wait_total_s": round(sum(wait_h2d), 3),
            "free_total_s": round(sum(free), 3),
            "prefetch_launch_total_s": round(sum(prefetch), 3),
            "sync_h2d_avg_ms": round(sum(sync_h2d) / max(len(sync_h2d), 1) * 1000, 2),
            "async_h2d_wait_avg_ms": round(sum(wait_h2d) / max(len(wait_h2d), 1) * 1000, 2),
            "free_avg_ms": round(sum(free) / max(len(free), 1) * 1000, 2),
            "prefetch_launch_avg_ms": round(sum(prefetch) / max(len(prefetch), 1) * 1000, 2),
        }


# ══════════════════════════════════════════════════════════════
# Variant C: Staged Pinned Buffer — 진정한 async DMA
# ══════════════════════════════════════════════════════════════

class AsyncPipelineHookStaged:
    """
    Double-buffered pinned staging buffer 기반 비동기 파이프라인.

    소량(~386MB × 2 = ~772MB)의 pinned 호스트 버퍼만 할당하고,
    각 레이어의 H2D를 다음과 같이 수행:
    1. CPU memcpy: pageable params → pinned buffer (~15ms)
    2. Async DMA: pinned buffer → GPU (non_blocking=True, ~45ms)

    DMA는 compute와 오버랩되므로 per-layer time ≈ max(compute, DMA) + staging overhead

    Prefetch를 _pre_forward에서 시작하여 compute 도중 DMA가 실행되도록 함.
    """

    def __init__(self):
        self.cpu_params = {}
        self.vlm_layers = None
        self.num_offload = 0

        self.transfer_stream = torch.cuda.Stream()
        self.prefetch_events = {}
        self.prefetch_active = {}

        # Double-buffered pinned staging
        self.pinned_bufs = [None, None]  # 두 개의 pinned buffer dict
        self.buf_idx = 0                 # 현재 사용할 buffer index

        self.transfer_times = []
        self.current_pass = -1
        self.hooks = []

    def save_cpu_refs(self, layers, num_offload):
        self.vlm_layers = layers
        self.num_offload = num_offload
        for i in range(num_offload):
            self.cpu_params[i] = {}
            for name, param in layers[i].named_parameters():
                self.cpu_params[i][name] = param.data
        print(f"    Saved CPU refs for {num_offload} layers")

    def allocate_pinned_buffers(self):
        """레이어 0의 파라미터 shape을 기반으로 double-buffered pinned buffer 할당"""
        t0 = time.time()
        total_bytes = 0
        for b in range(2):
            self.pinned_bufs[b] = {}
            for name, tensor in self.cpu_params[0].items():
                pinned = torch.empty_like(tensor, pin_memory=True)
                self.pinned_bufs[b][name] = pinned
                total_bytes += pinned.numel() * pinned.element_size()
        t1 = time.time()
        print(f"    Allocated 2 pinned buffers: {total_bytes / 1024**3:.3f} GB total in {t1-t0:.1f}s")

    def _stage_and_prefetch(self, layer_idx):
        """
        CPU memcpy(pageable→pinned) + async DMA(pinned→GPU).
        CPU가 memcpy 동안 block되지만, DMA는 transfer_stream에서 비동기 실행.
        """
        layer = self.vlm_layers[layer_idx]
        param_dict = dict(layer.named_parameters())
        buf = self.pinned_bufs[self.buf_idx]
        self.buf_idx = 1 - self.buf_idx  # 다음에는 다른 buffer 사용

        # Step 1: CPU memcpy — pageable → pinned (이 부분은 CPU blocking)
        t_copy0 = time.perf_counter()
        for name, cpu_tensor in self.cpu_params[layer_idx].items():
            if name in buf:
                buf[name].copy_(cpu_tensor)  # CPU memcpy
        t_copy1 = time.perf_counter()
        self.transfer_times.append(("STAGE_COPY", t_copy1 - t_copy0, layer_idx, self.current_pass))

        # Step 2: Async DMA — pinned → GPU (진정한 non_blocking)
        event = torch.cuda.Event()
        with torch.cuda.stream(self.transfer_stream):
            for name in self.cpu_params[layer_idx]:
                if name in param_dict and name in buf:
                    param_dict[name].data = buf[name].to("cuda", non_blocking=True)
            # 버퍼도 이동
            for bname, bval in layer.named_buffers():
                if bval.device.type == "cpu":
                    bval.data = bval.data.to("cuda", non_blocking=True)
            event.record()

        self.prefetch_events[layer_idx] = event
        self.prefetch_active[layer_idx] = True

    def _pre_forward(self, module, input, layer_idx):
        t0 = time.perf_counter()

        if layer_idx == 0:
            self.current_pass += 1
            self.prefetch_active.clear()
            self.prefetch_events.clear()

        if self.prefetch_active.get(layer_idx, False):
            # DMA 완료 대기 (GPU-side wait)
            torch.cuda.current_stream().wait_event(self.prefetch_events[layer_idx])
            self.prefetch_active[layer_idx] = False
            t1 = time.perf_counter()
            self.transfer_times.append(("WAIT_DMA", t1 - t0, layer_idx, self.current_pass))
        else:
            # 첫 레이어: 동기 H2D
            param_dict = dict(module.named_parameters())
            for name, cpu_tensor in self.cpu_params[layer_idx].items():
                if name in param_dict:
                    param_dict[name].data = cpu_tensor.to("cuda", non_blocking=False)
            for name, buf in module.named_buffers():
                if buf.device.type == "cpu":
                    buf.data = buf.data.to("cuda", non_blocking=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            self.transfer_times.append(("SYNC_H2D", t1 - t0, layer_idx, self.current_pass))

        # 다음 레이어 prefetch 시작 (compute와 DMA 오버랩)
        next_idx = layer_idx + 1
        if next_idx < self.num_offload:
            t_pf0 = time.perf_counter()
            self._stage_and_prefetch(next_idx)
            t_pf1 = time.perf_counter()
            self.transfer_times.append(("PREFETCH_TOTAL", t_pf1 - t_pf0, next_idx, self.current_pass))

    def _post_forward(self, module, input, output, layer_idx):
        """GPU 메모리 해제만 수행 (prefetch는 _pre_forward에서 이미 시작)"""
        t0 = time.perf_counter()
        param_dict = dict(module.named_parameters())
        for name, cpu_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict:
                param_dict[name].data = cpu_tensor
        for name, buf in module.named_buffers():
            if buf.device.type == "cuda":
                buf.data = buf.data.cpu()
        t1 = time.perf_counter()
        self.transfer_times.append(("FREE", t1 - t0, layer_idx, self.current_pass))

    def register_hooks(self, layers, num_offload):
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
        sync_h2d = [t for ty, t, li, p in self.transfer_times if ty == "SYNC_H2D"]
        wait_dma = [t for ty, t, li, p in self.transfer_times if ty == "WAIT_DMA"]
        stage_copy = [t for ty, t, li, p in self.transfer_times if ty == "STAGE_COPY"]
        prefetch_total = [t for ty, t, li, p in self.transfer_times if ty == "PREFETCH_TOTAL"]
        free = [t for ty, t, li, p in self.transfer_times if ty == "FREE"]
        return {
            "total_passes": self.current_pass + 1,
            "sync_h2d_count": len(sync_h2d),
            "wait_dma_count": len(wait_dma),
            "free_count": len(free),
            "stage_copy_count": len(stage_copy),
            "sync_h2d_total_s": round(sum(sync_h2d), 3),
            "wait_dma_total_s": round(sum(wait_dma), 3),
            "stage_copy_total_s": round(sum(stage_copy), 3),
            "prefetch_total_s": round(sum(prefetch_total), 3),
            "free_total_s": round(sum(free), 3),
            "sync_h2d_avg_ms": round(sum(sync_h2d) / max(len(sync_h2d), 1) * 1000, 2),
            "wait_dma_avg_ms": round(sum(wait_dma) / max(len(wait_dma), 1) * 1000, 2),
            "stage_copy_avg_ms": round(sum(stage_copy) / max(len(stage_copy), 1) * 1000, 2),
            "prefetch_total_avg_ms": round(sum(prefetch_total) / max(len(prefetch_total), 1) * 1000, 2),
            "free_avg_ms": round(sum(free) / max(len(free), 1) * 1000, 2),
        }


# ══════════════════════════════════════════════════════════════
# 공통 인프라 (exp 07 동일)
# ══════════════════════════════════════════════════════════════

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


def offload_layers_to_cpu(model, hook, num_offload=30):
    vlm_layers = model.vlm.model.language_model.layers
    total_layers = len(vlm_layers)
    print(f"  Offloading {num_offload}/{total_layers} VLM layers to CPU...")
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
    hook.save_cpu_refs(vlm_layers, num_offload)
    gpu_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  Final GPU: {gpu_gb:.2f} GB")
    return gpu_gb, num_offload


# ══════════════════════════════════════════════════════════════
# 실험 실행
# ══════════════════════════════════════════════════════════════

def run_variant(variant_name, hook_class, extra_setup=None):
    """단일 variant 실행"""
    print(f"\n{'='*70}")
    print(f"Experiment: {variant_name}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    result = {"variant": variant_name}
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Phase 1: 모델 로드
    print("\n[Phase 1] Loading model...")
    try:
        model = load_model_direct_cuda()
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback; traceback.print_exc()
        result["error"] = str(e)
        return result

    result["cuda_load_time"] = round(time.time() - t0, 2)
    print(f"  Load time: {result['cuda_load_time']}s")

    # Phase 2: 오프로드
    print("\n[Phase 2] Setting up pipeline...")
    t2 = time.time()
    hook = hook_class()

    try:
        gpu_resident, num_offloaded = offload_layers_to_cpu(model, hook, num_offload=30)
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback; traceback.print_exc()
        result["error"] = str(e)
        del model; torch.cuda.empty_cache(); gc.collect()
        return result

    result["setup_time"] = round(time.time() - t2, 2)
    result["gpu_resident_gb"] = round(gpu_resident, 2)
    result["vlm_layers_offloaded"] = num_offloaded

    # Extra setup (pinned buffer allocation 등)
    if extra_setup:
        extra_setup(hook)

    # Hook 등록
    vlm_layers = model.vlm.model.language_model.layers
    hook.register_hooks(vlm_layers, num_offloaded)
    print(f"  Hooks registered: {num_offloaded} offloaded layers")

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
    print(f"\n[Phase 4] Running inference ({variant_name})...")
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
        import traceback; traceback.print_exc()

    # Phase 5: 통계
    result["transfer_stats"] = hook.get_stats()
    result["total_time"] = round(t5 - t0, 2)

    stats = result["transfer_stats"]
    print(f"\n{'─'*50}")
    print(f"RESULTS: {variant_name}")
    print(f"{'─'*50}")
    print(f"  Inference: {result.get('inference_time', 'N/A')}s")
    print(f"  Peak VRAM: {result.get('peak_vram_gb', 'N/A')} GB")
    for key, val in stats.items():
        print(f"  {key}: {val}")

    hook.remove_hooks()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return result


def generate_visualizations(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline = {"inference_time": 43.38, "label": "Method 1\n(Sync H2D+Free)"}
    labels = [baseline["label"]]
    times = [baseline["inference_time"]]
    colors = ["#4A90D9"]

    for r in results:
        if r.get("error"):
            continue
        label = r["variant"].replace("_", "\n")
        labels.append(label)
        times.append(r["inference_time"])
        colors.append("#E8793A" if "pageable" in r["variant"] else "#5CB85C")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, times, color=colors, width=0.5, edgecolor="black", linewidth=0.5)

    for i, (bar, t) in enumerate(zip(bars, times)):
        speedup = baseline["inference_time"] / t
        text = f"{t}s\n({speedup:.2f}x)" if i > 0 else f"{t}s\n(baseline)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                text, ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Inference Time (s)", fontsize=12)
    ax.set_title("Async Pipeline vs Sync Baseline", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(times) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "async_pipeline_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: async_pipeline_comparison.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["a", "c", "both"], default="both")
    parser.add_argument("--visualize-only", action="store_true")
    args = parser.parse_args()

    if args.visualize_only:
        with open(os.path.join(RESULTS_DIR, "results.json")) as f:
            data = json.load(f)
        generate_visualizations(data["variants"])
        return

    print("=" * 70)
    print("ASYNC 2-STREAM PIPELINE EXPERIMENT (Method 4)")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    all_results = []

    if args.variant in ("a", "both"):
        result_a = run_variant("variant_a_pageable_async", AsyncPipelineHookPageable)
        all_results.append(result_a)

    if args.variant in ("c", "both"):
        def setup_pinned(hook):
            print("\n[Phase 2.5] Allocating pinned staging buffers...")
            hook.allocate_pinned_buffers()

        result_c = run_variant("variant_c_staged_pinned", AsyncPipelineHookStaged,
                               extra_setup=setup_pinned)
        all_results.append(result_c)

    # 종합
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    baseline_time = 43.38
    print(f"  Baseline (Method 1, sync): {baseline_time}s")
    for r in all_results:
        if r.get("error"):
            print(f"  {r['variant']}: FAILED — {r['error']}")
        else:
            speedup = baseline_time / r["inference_time"]
            print(f"  {r['variant']}: {r['inference_time']}s ({speedup:.2f}x vs baseline)")

    # 저장
    output = {
        "experiment": "async_2stream_pipeline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu_name,
        "baseline_method_1": {"inference_time": 43.38, "peak_vram_gb": 11.03,
                              "h2d_total_s": 24.226, "free_total_s": 1.045},
        "variants": all_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: results.json")

    print("\nGenerating visualizations...")
    try:
        generate_visualizations(all_results)
    except Exception as e:
        print(f"  Visualization failed: {e}")
    print("\nDone!")


if __name__ == "__main__":
    main()
