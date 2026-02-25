#!/usr/bin/env python3
"""Part 1: Alpamayo R1 Layer Structure Analysis

Loads the model on CPU and analyzes:
- Per-submodule layer count, parameter count, and memory size
- Breakdown by major components: vision_encoder, vlm (language model), expert, diffusion, etc.
"""

import sys
import os
import json
import time

# Add the alpamayo source to path
sys.path.insert(0, "/home/seungwoo/workspace/alpamayo/src")

import torch
from collections import OrderedDict

# ─── Helper functions ────────────────────────────────────────────────────────

def count_parameters(module):
    """Count total parameters and memory for a module."""
    total_params = 0
    total_bytes = 0
    for p in module.parameters():
        total_params += p.numel()
        total_bytes += p.numel() * p.element_size()
    return total_params, total_bytes

def format_bytes(nbytes):
    """Format bytes to human-readable string."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    elif nbytes >= 1e6:
        return f"{nbytes / 1e6:.2f} MB"
    elif nbytes >= 1e3:
        return f"{nbytes / 1e3:.2f} KB"
    return f"{nbytes} B"

def format_params(n):
    """Format parameter count."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)

def analyze_module_tree(module, prefix="", depth=0, max_depth=3):
    """Recursively analyze module tree up to max_depth."""
    results = []
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        params, mem = count_parameters(child)
        n_layers = sum(1 for _ in child.named_children())
        results.append({
            "name": full_name,
            "depth": depth,
            "params": params,
            "memory_bytes": mem,
            "n_children": n_layers,
        })
        if depth < max_depth:
            results.extend(analyze_module_tree(child, full_name, depth + 1, max_depth))
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Alpamayo R1 Layer Structure Analysis")
    print("=" * 80)

    # Load model on CPU with meta device first to get structure,
    # then load actual weights for accurate size measurement
    print("\n[1/4] Loading model on CPU (this may take a few minutes)...")
    t0 = time.time()

    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"    Model loaded in {load_time:.1f}s")

    # ── Overall statistics ──
    print("\n[2/4] Computing overall statistics...")
    total_params, total_mem = count_parameters(model)
    print(f"    Total parameters: {format_params(total_params)} ({total_params:,})")
    print(f"    Total memory (FP16): {format_bytes(total_mem)}")

    # ── Top-level submodule breakdown ──
    print("\n[3/4] Top-level submodule breakdown:")
    print("-" * 80)
    print(f"{'Submodule':<40} {'Params':>15} {'Memory (FP16)':>15} {'% of Total':>10}")
    print("-" * 80)

    top_level_data = OrderedDict()
    for name, child in model.named_children():
        params, mem = count_parameters(child)
        pct = (params / total_params * 100) if total_params > 0 else 0
        top_level_data[name] = {
            "params": params,
            "memory_bytes": mem,
            "pct": pct,
        }
        print(f"{name:<40} {format_params(params):>15} {format_bytes(mem):>15} {pct:>9.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<40} {format_params(total_params):>15} {format_bytes(total_mem):>15} {'100.0%':>10}")

    # ── Semantic grouping ──
    # Group into: Vision Encoder, VLM (Language Model), Expert (Diffusion Trajectory Decoder backbone),
    # Diffusion (flow matching), Action Projections, Tokenizers, Other
    print("\n\n[4/4] Semantic component breakdown:")
    print("=" * 80)

    # Identify submodules
    semantic_groups = OrderedDict()

    # VLM contains vision encoder + language model
    if hasattr(model, 'vlm'):
        vlm = model.vlm
        # Vision encoder is typically vlm.visual or vlm.model.visual
        if hasattr(vlm, 'visual'):
            ve_params, ve_mem = count_parameters(vlm.visual)
            semantic_groups["Vision Encoder (vlm.visual)"] = {"params": ve_params, "memory_bytes": ve_mem}
        elif hasattr(vlm, 'model') and hasattr(vlm.model, 'visual'):
            ve_params, ve_mem = count_parameters(vlm.model.visual)
            semantic_groups["Vision Encoder (vlm.model.visual)"] = {"params": ve_params, "memory_bytes": ve_mem}

        # Language model part
        if hasattr(vlm, 'model'):
            vlm_model = vlm.model
            lm_params, lm_mem = count_parameters(vlm_model)
            # Subtract vision encoder if it's inside vlm.model
            if hasattr(vlm_model, 'visual'):
                ve_p, ve_m = count_parameters(vlm_model.visual)
                lm_params -= ve_p
                lm_mem -= ve_m
            semantic_groups["VLM Language Model (vlm.model - visual)"] = {"params": lm_params, "memory_bytes": lm_mem}

        # LM head
        if hasattr(vlm, 'lm_head'):
            lmh_params, lmh_mem = count_parameters(vlm.lm_head)
            semantic_groups["VLM LM Head"] = {"params": lmh_params, "memory_bytes": lmh_mem}

    # Expert model (Diffusion Trajectory Decoder backbone)
    if hasattr(model, 'expert'):
        exp_params, exp_mem = count_parameters(model.expert)
        semantic_groups["Expert (Trajectory Decoder Backbone)"] = {"params": exp_params, "memory_bytes": exp_mem}

    # Diffusion (flow matching)
    if hasattr(model, 'diffusion'):
        diff_params, diff_mem = count_parameters(model.diffusion)
        semantic_groups["Diffusion (Flow Matching)"] = {"params": diff_params, "memory_bytes": diff_mem}

    # Action projections
    for proj_name in ['action_in_proj', 'action_out_proj']:
        if hasattr(model, proj_name):
            p, m = count_parameters(getattr(model, proj_name))
            semantic_groups[f"Action Projection ({proj_name})"] = {"params": p, "memory_bytes": m}

    # Trajectory tokenizers
    for tok_name in ['traj_tokenizer', 'hist_traj_tokenizer']:
        mod = getattr(model, tok_name, None)
        if mod is not None and isinstance(mod, torch.nn.Module):
            p, m = count_parameters(mod)
            if p > 0:
                semantic_groups[f"Trajectory Tokenizer ({tok_name})"] = {"params": p, "memory_bytes": m}

    print(f"{'Component':<50} {'Params':>15} {'Memory (FP16)':>15} {'% of Total':>10}")
    print("-" * 90)

    accounted_params = 0
    for comp_name, info in semantic_groups.items():
        pct = (info["params"] / total_params * 100) if total_params > 0 else 0
        print(f"{comp_name:<50} {format_params(info['params']):>15} {format_bytes(info['memory_bytes']):>15} {pct:>9.1f}%")
        accounted_params += info["params"]

    unaccounted = total_params - accounted_params
    if unaccounted > 0:
        unaccounted_mem = unaccounted * 2  # FP16
        pct = unaccounted / total_params * 100
        print(f"{'Other/Unaccounted':<50} {format_params(unaccounted):>15} {format_bytes(unaccounted_mem):>15} {pct:>9.1f}%")

    print("-" * 90)
    print(f"{'TOTAL':<50} {format_params(total_params):>15} {format_bytes(total_mem):>15} {'100.0%':>10}")

    # ── Detailed layer-by-layer for expert ──
    print("\n\nDetailed: Expert model layers")
    print("-" * 80)
    if hasattr(model, 'expert'):
        for name, child in model.expert.named_children():
            p, m = count_parameters(child)
            print(f"  expert.{name:<35} {format_params(p):>15} {format_bytes(m):>15}")
            # One more level
            for name2, child2 in child.named_children():
                p2, m2 = count_parameters(child2)
                if p2 > 0:
                    n_sub = sum(1 for _ in child2.named_children())
                    layer_info = f" ({n_sub} sublayers)" if n_sub > 0 else ""
                    print(f"    expert.{name}.{name2:<30} {format_params(p2):>15} {format_bytes(m2):>15}{layer_info}")

    # ── Detailed: VLM model layers (just first 2 levels) ──
    print("\n\nDetailed: VLM model structure (depth 2)")
    print("-" * 80)
    if hasattr(model, 'vlm') and hasattr(model.vlm, 'model'):
        for name, child in model.vlm.model.named_children():
            p, m = count_parameters(child)
            n_sub = sum(1 for _ in child.named_children())
            print(f"  vlm.model.{name:<35} {format_params(p):>15} {format_bytes(m):>15} ({n_sub} children)")

    # ── VLM transformer layers count ──
    print("\n\nVLM Transformer Layer Details:")
    print("-" * 80)
    if hasattr(model, 'vlm') and hasattr(model.vlm, 'model'):
        vlm_model = model.vlm.model
        # Check for layers attribute (common in transformers)
        layers_attr = None
        for attr_name in ['layers', 'decoder', 'encoder']:
            if hasattr(vlm_model, attr_name):
                candidate = getattr(vlm_model, attr_name)
                if hasattr(candidate, '__len__'):
                    layers_attr = (attr_name, candidate)
                    break
                elif hasattr(candidate, 'layers'):
                    layers_attr = (f"{attr_name}.layers", candidate.layers)
                    break

        if layers_attr:
            attr_name, layers = layers_attr
            n_layers = len(layers)
            print(f"  Transformer layers location: vlm.model.{attr_name}")
            print(f"  Number of transformer layers: {n_layers}")
            if n_layers > 0:
                first_layer = layers[0]
                lp, lm = count_parameters(first_layer)
                print(f"  Parameters per layer: {format_params(lp)} ({format_bytes(lm)})")
                print(f"  Total transformer layers: {format_params(lp * n_layers)} ({format_bytes(lm * n_layers)})")

    # ── Save results to JSON ──
    results = {
        "total_params": total_params,
        "total_memory_bytes_fp16": total_mem,
        "load_time_seconds": load_time,
        "top_level_modules": {
            name: {"params": info["params"], "memory_bytes": info["memory_bytes"], "pct": round(info["pct"], 2)}
            for name, info in top_level_data.items()
        },
        "semantic_groups": {
            name: {"params": info["params"], "memory_bytes": info["memory_bytes"],
                   "pct": round(info["params"] / total_params * 100, 2)}
            for name, info in semantic_groups.items()
        },
    }

    output_path = "/home/seungwoo/workspace/research/01-on-demand-layering/layer_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")

    # Cleanup
    del model
    import gc
    gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
