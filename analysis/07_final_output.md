# Stage 7: 최종 출력 및 전체 요약

## 개요

다중 trajectory 샘플링 결과를 정리하여 최종 예측 출력.
`num_traj_samples`개의 후보 trajectory 생성 후 minADE 기준으로 평가.

**코드 위치**: `alpamayo_r1.py` L291–317

---

## 출력 형태

**위치**: `alpamayo_r1.py` L311–317

```
pred_xyz: (B, num_traj_sets, num_traj_samples, 64, 3)
pred_rot: (B, num_traj_sets, num_traj_samples, 64, 3, 3)
```

| 차원 | 의미 | 기본값 |
|------|------|--------|
| `B` | batch size | — |
| `num_traj_sets` | trajectory set 수 | 1 |
| `num_traj_samples` | 샘플 수 | 6 |
| `64` | waypoint 수 | 고정 |
| `3` | xyz 좌표 | 고정 |
| `3, 3` | SO(3) 회전 행렬 | 고정 |

---

## 다중 샘플링 구조

```
Stage 5 Diffusion (Flow Matching)
    ↓ × num_traj_samples (=6)

초기 노이즈 6개 독립 샘플링
    ↓
각각 10 Euler 스텝 → 6개 denoised action
    ↓
Stage 6 action_to_traj × 6
    ↓
pred_xyz: (B, 1, 6, 64, 3)
```

---

## 평가 메트릭: minADE

**Minimum Average Displacement Error**

```
ADE_k = (1/64) × Σ_{t=1}^{64} ||pred_xyz[k, t] - gt_xyz[t]||₂

minADE = min_{k=1..6} ADE_k
```

- 6개 후보 중 ground truth에 가장 가까운 trajectory 선택
- 자율주행의 다중 모달 미래 분포를 커버리지 관점에서 평가
- 1개 샘플 ADE 대비 낮은 오차 달성

---

## 전체 파이프라인 요약

| 단계 | 모듈 | 입력 | 출력 | 역할 |
|------|------|------|------|------|
| Stage 1 | Patch Embedding | Raw image (B, T, C, H, W) | (N_patches, 1152) | 3D Conv + 3D RoPE |
| Stage 2 | Vision Encoder | (N_patches, 1152) | (N_patches, 4096) | 32-layer ViT (SigLIP) |
| Stage 3 | Patch Merger | (N_patches, 4096) | (N_patches/4, 4096) | Spatial 2×2 merge + MLP |
| Stage 4 | VLM Generation | Vision+Text tokens | KV cache (36 layers) | Qwen3-VL-8B CoT 추론 |
| Stage 5 | Diffusion Sampling | Noise (B, 64, 2) | Denoised (B, 64, 2) | Flow Matching 10 steps |
| Stage 6 | Action → Traj | Action (B, 64, 2) | xyz (B, 64, 3), rot (B, 64, 3, 3) | Unicycle 적분 |
| Stage 7 | Final Output | 6× traj 샘플 | (B, 1, 6, 64, 3) | 다중 샘플 취합 |

---

## 파라미터 수 요약

| 모듈 | 파라미터 수 | 비고 |
|------|-----------|------|
| Vision Encoder | ~400M | SigLIP-400M backbone |
| VLM (Qwen3-VL-8B) | ~8B | 36-layer Transformer |
| Expert Decoder | ~2B | text_config 기반, 16 layers |
| Action In/Out Proj | ~5M | Fourier + MLP |
| Flow Matching | ~5M | 경량 네트워크 |
| **전체** | **~10.4B** | |

---

## 추론 시간 분석

연구 결과 참조 (`research/` 디렉토리):

| 방법 | 추론 시간 | 베이스라인 대비 |
|------|----------|--------------|
| 베이스라인 (FP16, CUDA Unified Memory) | 273.79s | 1× |
| Demand Layering (D2H 제거) | 43.38s | **6.31× 개선** |

### 핵심 발견
- D2H (Device-to-Host) 복사 불필요 — CPU 원본 유지로 제거 가능
- WSL2 Pageable D2H: 비정상 동작 확인
- 최적 청크 크기: 2–8 MB

---

## 관련 파일

- [`figures/pipeline_overview.png`](figures/pipeline_overview.png) — 전체 파이프라인 다이어그램
- [`figures/architecture_overview.png`](figures/architecture_overview.png) — 아키텍처 개요
- `alpamayo_r1.py` L291–317 — 최종 출력 조립
- `alpamayo_r1.py` L122–328 — `sample_trajectories_from_data_with_vlm_rollout` 전체
