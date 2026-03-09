"""
Pinned Memory 실험 v2 — On-Demand Layering with Pinned Memory

device_map="auto"로 모델 로드 후 오프로드된 CPU 레이어를 pinned memory로 변환.
Forward hook으로 on-demand H2D 전송 (non_blocking=True).

비교 대상: baseline (device_map=auto, 184.99s)

변경점:
- 모델 로드: device_map="auto" (baseline과 동일)
- CPU 레이어를 pin_memory()로 고정
- pre_forward: pinned CPU → GPU (H2D, non_blocking)
- post_forward: GPU 파라미터를 pinned CPU 원본으로 교체 (D2H 없음)
"""
import sys
import os
import time
import json
import gc
import threading
import subprocess

sys.path.insert(0, "/home/avees/workspace/alpamayo/src")

import torch
import torch.nn as nn

HF_CHECKPOINT = "/home/avees/.cache/huggingface/hub/models--nvidia--Alpamayo-R1-10B/snapshots/22fab1399111f50b52bfbe5d8b809f39bd4c2fe1"
RESULTS_DIR = "/home/avees/workspace"


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


class LayerOffloadHookPinned:
    """
    Pinned Memory 기반 레이어 오프로드 훅.

    device_map="auto"로 이미 CPU에 배치된 레이어를 pin_memory()로 변환.
    - pre_forward: pinned CPU → GPU (H2D, non_blocking)
    - post_forward: GPU 파라미터를 pinned CPU 원본으로 교체 (D2H 없음)
    """

    def __init__(self):
        self.cpu_params = {}   # layer_idx -> {name: pinned_cpu_tensor}
        self.cpu_buffers = {}  # layer_idx -> {name: pinned_cpu_tensor}
        self.transfer_times = []
        self.hooks = []
        self.pin_time = 0.0

    def convert_to_pinned(self, layers: nn.ModuleList, cpu_layer_indices: list):
        """CPU 레이어를 pinned memory로 변환"""
        t0 = time.perf_counter()
        total_params = 0
        total_bytes = 0

        for i in cpu_layer_indices:
            layer = layers[i]
            self.cpu_params[i] = {}
            self.cpu_buffers[i] = {}

            for name, param in layer.named_parameters():
                if param.device.type == "cpu":
                    if not param.data.is_pinned():
                        pinned = param.data.pin_memory()
                    else:
                        pinned = param.data
                    self.cpu_params[i][name] = pinned
                    param.data = pinned
                    total_params += 1
                    total_bytes += pinned.nelement() * pinned.element_size()

            for name, buf in layer.named_buffers():
                if buf.device.type == "cpu":
                    if not buf.data.is_pinned():
                        pinned_buf = buf.data.pin_memory()
                    else:
                        pinned_buf = buf.data
                    self.cpu_buffers[i][name] = pinned_buf
                    buf.data = pinned_buf

        t1 = time.perf_counter()
        self.pin_time = t1 - t0
        total_gb = total_bytes / 1024**3
        print(f"    Pinned {len(cpu_layer_indices)} layers ({total_params} params, {total_gb:.2f} GB)")
        print(f"    Pinning time: {self.pin_time:.2f}s")

    def _pre_forward(self, module, input, layer_idx):
        """H2D: Pinned CPU → GPU (non_blocking 활용)"""
        t0 = time.perf_counter()

        param_dict = dict(module.named_parameters())
        for name, pinned_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict and param_dict[name].device.type == "cpu":
                param_dict[name].data = pinned_tensor.to("cuda", non_blocking=True)

        buf_dict = dict(module.named_buffers())
        for name, pinned_buf in self.cpu_buffers[layer_idx].items():
            if name in buf_dict and buf_dict[name].device.type == "cpu":
                buf_dict[name].data = pinned_buf.to("cuda", non_blocking=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        self.transfer_times.append(("H2D", t1 - t0))

    def _post_forward(self, module, input, output, layer_idx):
        """GPU Free: Pinned CPU 원본 복원 (D2H 없음)"""
        t0 = time.perf_counter()

        param_dict = dict(module.named_parameters())
        for name, pinned_tensor in self.cpu_params[layer_idx].items():
            if name in param_dict:
                param_dict[name].data = pinned_tensor

        buf_dict = dict(module.named_buffers())
        for name, pinned_buf in self.cpu_buffers[layer_idx].items():
            if name in buf_dict:
                buf_dict[name].data = pinned_buf

        torch.cuda.empty_cache()
        t1 = time.perf_counter()
        self.transfer_times.append(("FREE", t1 - t0))

    def register_hooks(self, layers: nn.ModuleList, cpu_layer_indices: list):
        for i in cpu_layer_indices:
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


def get_ram_usage():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_experiment():
    """Pinned Memory On-Demand Layering 실험 v2"""
    config_name = "pinned_memory_demand_layering_v2"

    print(f"\n{'='*70}")
    print(f"Experiment: {config_name}")
    print(f"{'='*70}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_vram_gb:.1f} GB")
    print(f"GPU Info: {get_gpu_info()}")

    result = {"config": config_name, "gpu": torch.cuda.get_device_name(0)}
    monitor = VRAMMonitor(interval=0.1)
    monitor.start()

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # Phase 1: 모델 로드 (device_map="auto")
    print("\n[Phase 1] Loading model with device_map='auto'...")
    model_load_start = time.time() - monitor.t0

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

    try:
        model = AlpamayoR1.from_pretrained(
            HF_CHECKPOINT,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback; traceback.print_exc()
        monitor.stop()
        return {"error": str(e)}

    t_load = time.time()
    result["cuda_load_time"] = round(t_load - t0, 2)
    result["model_load_start_t"] = model_load_start
    result["model_load_end_t"] = time.time() - monitor.t0

    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"  Load time: {result['cuda_load_time']}s | VRAM: {vram_after_load:.2f} GB")

    # 레이어 배치 확인
    vlm_layers = model.vlm.model.language_model.layers
    total_layers = len(vlm_layers)
    cpu_layer_indices = []
    gpu_layer_indices = []
    for i, layer in enumerate(vlm_layers):
        # 첫 파라미터의 device 확인
        try:
            first_param = next(layer.parameters())
            if first_param.device.type == "cpu":
                cpu_layer_indices.append(i)
            else:
                gpu_layer_indices.append(i)
        except StopIteration:
            pass

    print(f"  VLM layers total: {total_layers}")
    print(f"  CPU layers: {len(cpu_layer_indices)} ({cpu_layer_indices[:5]}...)")
    print(f"  GPU layers: {len(gpu_layer_indices)}")
    result["vlm_layers_cpu"] = len(cpu_layer_indices)
    result["vlm_layers_gpu"] = len(gpu_layer_indices)

    # Phase 2: CPU 레이어를 Pinned Memory로 변환
    print("\n[Phase 2] Converting CPU layers to pinned memory...")
    t2 = time.time()
    hook = LayerOffloadHookPinned()

    if not cpu_layer_indices:
        print("  [INFO] 모든 레이어가 GPU에 있음. 오프로드 불필요.")
        # 수동으로 레이어 오프로드
        num_offload = 25  # 5070 Ti 16GB에서 적절한 오프로드 수
        print(f"  수동으로 {num_offload}개 레이어를 CPU로 이동...")
        for i in range(num_offload):
            vlm_layers[i].to("cpu")
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"    Layer {i+1}/{num_offload} | GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        cpu_layer_indices = list(range(num_offload))

    try:
        hook.convert_to_pinned(vlm_layers, cpu_layer_indices)
    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback; traceback.print_exc()
        monitor.stop()
        del model; torch.cuda.empty_cache(); gc.collect()
        return {"error": str(e)}

    result["setup_time"] = round(time.time() - t2, 2)
    result["gpu_resident_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    result["vlm_layers_offloaded"] = len(cpu_layer_indices)

    # Hook 등록
    hook.register_hooks(vlm_layers, cpu_layer_indices)
    print(f"  Hooks registered for {len(cpu_layer_indices)} layers")
    print(f"  GPU after setup: {result['gpu_resident_gb']:.2f} GB")

    # Phase 3: 입력 준비
    print("\n[Phase 3] Loading dataset & preparing inputs...")
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
    data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
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
    print("\n[Phase 4] Running inference (pinned H2D + free)...")
    print(f"  GPU Info (before inference): {get_gpu_info()}")
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
    result["total_time"] = round(t5 - t0, 2)

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
    print("RESULTS (Pinned Memory v2)")
    print(f"{'='*70}")
    stats = result.get("transfer_stats", {})
    print(f"  Load: {result.get('cuda_load_time')}s | Setup: {result.get('setup_time')}s")
    print(f"  Inference: {result.get('inference_time', 'N/A')}s")
    print(f"  Peak VRAM: {result.get('peak_vram_gb', 'N/A')} GB")
    if stats:
        print(f"  H2D: {stats.get('h2d_total_s')}s ({stats.get('h2d_avg_ms')}ms avg)")
        print(f"  FREE: {stats.get('free_total_s')}s ({stats.get('free_avg_ms')}ms avg)")
        print(f"  Pin time: {stats.get('pin_time_s')}s")

    # 결과 저장
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
    print("Alpamayo-R1-10B — Pinned Memory On-Demand Layering v2")
    print("=" * 70)
    run_experiment()
