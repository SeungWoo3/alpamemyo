# Alpamayo-R1-10B 추론 파이프라인 분석

## 프로젝트 개요

**목적**: Alpamayo-R1-10B 모델의 추론 파이프라인 전체 구조를 단계별로 분석하여, On-Demand Layering 최적화의 기초 자료로 활용

**대상 모델**: NVIDIA Alpamayo-R1-10B — 자율주행용 Vision-Language-Action(VLA) 모델
- Vision Encoder: Qwen3-VL 기반 ViT (27 레이어, hidden=1152)
- VLM 백본: Qwen3-VL-8B (36 레이어, hidden=4096)
- Expert Decoder: 경량 Transformer (hidden=2048)
- 액션 생성: Flow Matching 기반 Diffusion (10 Euler 스텝)

---

## 7-Stage 파이프라인 요약

| 단계 | 이름 | 입력 | 출력 | 핵심 연산 |
|------|------|------|------|-----------|
| Stage 1 | [Patch Embedding](01_patch_embedding.md) | `(B, 3, T, H, W)` 이미지 텐서 | `(N_patches, 1152)` | Conv3d + 3D RoPE |
| Stage 2 | [Vision Transformer](02_vision_transformer.md) | `(N_patches, 1152)` | `(N_patches, 1152)` | 27× LayerNorm→MHA→MLP |
| Stage 3 | [Patch Merger](03_patch_merger.md) | `(N_patches, 1152)` | `(N_patches/4, 4096)` | 2×2 Concat → Linear(4608→4096) |
| Stage 4 | VLM Prefill | `(N_vis + N_text, 4096)` | `(N_seq, 4096)` | 36× GQA Decoder Layer |
| Stage 5 | VLM Decode | `(1, 4096)` + KV Cache | `(N_reason, 4096)` | Autoregressive 생성 |
| Stage 6 | Expert Decode | `(N_reason, 4096)` → `(N_act, 2048)` | `(N_act, 2048)` | Action projection + Decoder |
| Stage 7 | Diffusion | Noisy waypoints | 64 waypoints (accel, curvature) | Flow Matching, 10 Euler 스텝 |

---

## 핵심 수치 요약

### Vision Encoder
| 항목 | 값 |
|------|----|
| Hidden dimension | 1152 |
| Attention heads | 16 |
| Head dimension | 72 (= 1152 / 16) |
| Transformer 레이어 수 | 27 |
| Patch 크기 | 16×16 (spatial), 2-frame (temporal) |
| Conv3d 커널 | (2, 16, 16), stride=(2, 16, 16) |
| 입력 채널 | 3 (RGB) |
| MLP 확장 비율 | 4304 / 1152 ≈ 3.73 |
| 활성화 함수 | SiLU |

### Patch Merger
| 항목 | 값 |
|------|----|
| 공간 병합 크기 | 2×2 |
| 입력 차원 | 1152 × 4 = 4608 |
| 출력 차원 | 4096 |
| 활성화 함수 | GELU |

### VLM (Qwen3-VL-8B)
| 항목 | 값 |
|------|----|
| Hidden dimension | 4096 |
| Intermediate dimension | 12288 |
| 레이어 수 | 36 |
| Attention heads (Q) | 32 |
| KV heads (GQA) | 8 |
| Head dimension | 128 |

### Expert Decoder
| 항목 | 값 |
|------|----|
| Hidden dimension | 2048 |
| Intermediate dimension | 8256 |
| Attention heads | 16 |
| Head dimension | 128 |

### Action Space
| 항목 | 값 |
|------|----|
| Waypoint 수 | 64 |
| 액션 차원 | 2 (accel, curvature) |
| 시간 간격 (dt) | 0.1 s |
| 총 예측 수평선 | 6.4 s |

### Diffusion
| 항목 | 값 |
|------|----|
| 방식 | Flow Matching |
| Euler 스텝 수 | 10 |
| 노이즈 스케줄 | Linear (t: 0→1) |

---

## 단계별 문서

| 문서 | 내용 |
|------|------|
| [01_patch_embedding.md](01_patch_embedding.md) | Conv3d 패치 임베딩 + 3D Rotary Embedding |
| [02_vision_transformer.md](02_vision_transformer.md) | 27-Layer Vision Transformer 블록 구조 |
| [03_patch_merger.md](03_patch_merger.md) | 2×2 패치 병합 + VLM 차원 투영 |

---

## 시각화 자료 목록

| 파일 | 내용 |
|------|------|
| [figures/vision_encoder_detail.png](figures/vision_encoder_detail.png) | Vision Encoder 전체 구조 (PatchEmbed → 27 Blocks → Merger) |

---

## 코드 참조 목록

| 파일 | 내용 |
|------|------|
| [code_references/vision_encoder.md](code_references/vision_encoder.md) | `Qwen3VLVisionPatchEmbed`, `Qwen3VLVisionBlock`, `Qwen3VLVisionPatchMerger` 발췌 |
| [code_references/vlm_model.md](code_references/vlm_model.md) | VLM (Qwen3VLTextModel) 구조 발췌 |
| [code_references/action_in_proj.md](code_references/action_in_proj.md) | Action input projection 발췌 |
| [code_references/action_space.md](code_references/action_space.md) | Unicycle action space 발췌 |
| [code_references/diffusion.md](code_references/diffusion.md) | Flow Matching diffusion 발췌 |
| [code_references/inference.md](code_references/inference.md) | 추론 전체 흐름 발췌 |

---

## 소스 파일 위치

| 컴포넌트 | 경로 |
|----------|------|
| Vision Encoder | `alpamayo/ar1_venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py` |
| VLM 모델 | `alpamayo/src/alpamayo_r1/models/alpamayo_r1.py` |
| 베이스 모델 | `alpamayo/src/alpamayo_r1/models/base_model.py` |
| Diffusion | `alpamayo/src/alpamayo_r1/diffusion/flow_matching.py` |
| Action Space | `alpamayo/src/alpamayo_r1/action_space/unicycle_accel_curvature.py` |
| Action Projection | `alpamayo/src/alpamayo_r1/models/action_in_proj.py` |
| Config | `alpamayo/src/alpamayo_r1/config.py` |
