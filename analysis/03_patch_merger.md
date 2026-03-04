# Stage 3: Patch Merger

## 개요

Vision Transformer 출력 패치 시퀀스를 2×2 공간 단위로 병합하여 VLM의 텍스트 토큰과 동일한 차원(4096)으로 투영하는 단계.
패치 수를 1/4로 줄여 VLM 입력 시퀀스 길이를 단축, 계산 비용 절감.

---

## 입력

```
hidden_states: (N_patches, 1152)  — Stage 2 Vision Transformer 27번째 레이어 출력
```

N_patches: 공간적으로 배열된 패치의 전체 수 (예: T/2 × H/16 × W/16)

---

## 핵심 연산: `Qwen3VLVisionPatchMerger`

**소스**: `modeling_qwen3_vl.py` L93–106

```python
# L93-106
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        # spatial_merge_size=2 → 2×2=4 패치 병합
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        # = 1152 × 4 = 4608
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)  # (4608, 4608)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)  # (4608, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_patches, 1152) 이미 2×2 블록 순서로 재배열된 상태
        x = self.norm(
            x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x
        ).view(-1, self.hidden_size)
        # x: (N_patches/4, 4608)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        # x: (N_patches/4, 4096)
        return x
```

---

## 2×2 패치 병합 상세

### 병합 선행 단계: `rot_pos_emb` 내 공간 재배열

**소스**: `modeling_qwen3_vl.py` L603–640

실제 2×2 병합은 `forward`의 `.view(-1, self.hidden_size)` 호출 직전에 완성됨.
패치 순서는 `rot_pos_emb`에서 이미 2×2 블록 단위로 인터리빙 정렬:

```python
# L614-632 (rot_pos_emb 내부)
for num_frames, height, width in grid_thw:
    merged_h, merged_w = height // merge_size, width // merge_size  # merge_size=2

    block_rows = torch.arange(merged_h)  # 블록 행 인덱스
    block_cols = torch.arange(merged_w)  # 블록 열 인덱스
    intra_row = torch.arange(merge_size) # 블록 내 행 오프셋 [0, 1]
    intra_col = torch.arange(merge_size) # 블록 내 열 오프셋 [0, 1]

    # 전체 해상도 위치 계산 → 2×2 블록 단위로 순서 정렬
    row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
    col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
```

`fast_pos_embed_interpolate` (L642–701)도 동일 블록 패턴으로 패치 재배열:

```python
# L692-700
pos_embed = (
    pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
    .permute(0, 1, 3, 2, 4, 5)   # 블록 우선 순서로 재배열
    .flatten(0, 4)
)
```

### 병합 시 shape 변환

```
입력:  (N_patches, 1152)
         │
         ▼
.view(-1, 4608)  →  (N_patches/4, 4608)
# 인접 4개 패치(2×2 블록)의 feature를 채널 축으로 concat
```

---

## MLP 투영 상세

### 데이터 흐름

```
(N_patches/4, 4608)
  │
  ├─ LayerNorm(4608)     ← use_postshuffle_norm=False 시 1152로 먼저 norm (단, view 후 적용)
  │
  ├─ Linear(4608→4608)   → linear_fc1
  ├─ GELU()
  ├─ Linear(4608→4096)   → linear_fc2
  │
  ▼
(N_patches/4, 4096)
```

**주의**: 최종 Merger (`use_postshuffle_norm=False`)는 `view(-1, 4608)` 후 LayerNorm(4608) 적용.
DeepStack용 Merger (`use_postshuffle_norm=True`)는 LayerNorm(1152) 먼저 적용 후 view.

### 파라미터

| 레이어 | Shape | FP16 크기 |
|--------|-------|-----------|
| LayerNorm | (4608,) | 0.018 MB |
| linear_fc1 | (4608, 4608) | 42.5 MB |
| linear_fc2 | (4608, 4096) | 37.7 MB |
| **합계** | | **~80.2 MB** |

---

## VisionModel.forward 내 Merger 호출 위치

**소스**: `modeling_qwen3_vl.py` L738–753

```python
# L738-753
deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, ...)

    # DeepStack: 지정 레이어에서 별도 Merger (use_postshuffle_norm=True) 적용
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[...](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)

# 최종 Merger: 27번째 레이어 출력 처리
hidden_states = self.merger(hidden_states)   # L751
# hidden_states: (N_patches/4, 4096)

return hidden_states, deepstack_feature_lists
```

---

## 출력

```
hidden_states: (N_patches/4, 4096)
```

- 텍스트 토큰 임베딩 차원(4096)과 동일 → VLM 입력으로 직접 연결 가능
- 패치 수 1/4 감소로 VLM 시퀀스 길이 대폭 단축

예시 (224×224, 4 프레임 기준):
```
입력:  N_patches = 392  →  출력: 392/4 = 98 토큰
이미지 1장이 VLM에서 98개 토큰으로 표현
```

---

## 의의

| 항목 | 내용 |
|------|------|
| 차원 정합 | Vision Encoder(1152) → VLM(4096) 차원 브릿징 |
| 시퀀스 압축 | 패치 수 1/4 감소 → VLM 계산 비용 절감 |
| DeepStack 지원 | 중간 레이어용 별도 Merger로 다중 스케일 시각 특징 제공 |

---

## DeepStack Merger와의 비교

| 항목 | 최종 Merger | DeepStack Merger |
|------|-------------|-----------------|
| `use_postshuffle_norm` | `False` | `True` |
| LayerNorm 입력 차원 | 1152 (view 전) | 4608 (view 후) |
| 적용 레이어 | 27번째 (마지막) | `deepstack_visual_indexes`에 해당하는 중간 레이어 |
| 출력 용도 | VLM 시각 토큰 | VLM early layer hidden state 가중 합산 |

---

## 파이프라인 내 위치

```
Stage 2: Vision Transformer (27 Blocks)
  ↓ (N_patches, 1152)
Stage 3: Patch Merger ← 현재 단계
  ↓ (N_patches/4, 4096)
Stage 4: VLM Prefill (텍스트 토큰과 concat)
```

---

## 관련 파일

- 소스 코드: `modeling_qwen3_vl.py` (L93–106, L564–588, L738–753)
  - 전체 경로: `alpamayo/ar1_venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- 코드 참조: [code_references/vision_encoder.md](code_references/vision_encoder.md)
