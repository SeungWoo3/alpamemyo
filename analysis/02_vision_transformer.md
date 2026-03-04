# Stage 2: Vision Transformer

## 개요

27개 Transformer 레이어를 순차 통과하며 시각 특징을 추출하는 단계.
각 레이어는 Pre-Norm + Multi-Head Self-Attention + SiLU-MLP 구조.
3D Rotary Positional Embedding이 Attention Q/K에 적용되어 공간·시간 위치 정보 보존.

---

## 입력

```
hidden_states: (N_patches, 1152)  — Stage 1 Patch Embedding 출력
rotary_pos_emb: (N_patches, 72)   — Stage 1에서 생성된 RoPE 벡터
cu_seqlens: (N_images + 1,)       — 배치 내 각 이미지 경계 (Flash Attention용 cumsum)
```

---

## 레이어 구조: `Qwen3VLVisionBlock`

**소스**: `modeling_qwen3_vl.py` L251–275

```python
# L251-275
class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)   # hidden_size=1152
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```

### 단일 레이어 데이터 흐름

```
입력: x  (N_patches, 1152)
  │
  ├─ LayerNorm(1152) → norm1_x
  ├─ VisionAttention(norm1_x, RoPE) → attn_out  (N_patches, 1152)
  ├─ Residual: x = x + attn_out
  │
  ├─ LayerNorm(1152) → norm2_x
  ├─ VisionMLP(norm2_x) → mlp_out  (N_patches, 1152)
  └─ Residual: x = x + mlp_out
출력: x  (N_patches, 1152)
```

---

## Attention: `Qwen3VLVisionAttention`

**소스**: `modeling_qwen3_vl.py` L168–248

### 아키텍처

| 항목 | 값 |
|------|----|
| 입력 차원 (`dim`) | 1152 |
| 헤드 수 (`num_heads`) | 16 |
| 헤드 차원 (`head_dim`) | 72 (= 1152 / 16) |
| QKV projection | `Linear(1152, 1152 × 3 = 3456)` — 단일 행렬 |
| Output projection | `Linear(1152, 1152)` |
| Scaling factor | `head_dim ** -0.5 = 72 ** -0.5 ≈ 0.1178` |
| KV 그룹 수 | 1 (full MHA, GQA 아님) |
| Causal mask | False (Vision Encoder는 양방향) |

```python
# L168-178
class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size          # 1152
        self.num_heads = config.num_heads      # 16
        self.head_dim = self.dim // self.num_heads  # 72
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)  # (1152, 3456)
        self.proj = nn.Linear(self.dim, self.dim)                 # (1152, 1152)
        self.scaling = self.head_dim**-0.5     # ≈ 0.1178
```

### Forward 흐름

```python
# L182-248 (요약)
def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
            position_embeddings=None, **kwargs):
    seq_length = hidden_states.shape[0]  # N_patches

    # 1. QKV 분리: (N, 3*1152) → 3 × (N, 16, 72)
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)  # (N, 3, 16, 72)
        .permute(1, 0, 2, 3)                          # (3, N, 16, 72)
        .unbind(0)
    )

    # 2. RoPE 적용
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    # 3. Scaled Dot-Product Attention (SDPA / Flash Attention)
    # cu_seqlens로 이미지 경계 마스킹
    attn_output = ...  # (N_patches, 1152)

    # 4. Output projection
    attn_output = self.proj(attn_output)
    return attn_output
```

### 메모리 비용 (단일 레이어, N_patches=392)

| 텐서 | Shape | FP16 크기 |
|------|-------|-----------|
| QKV weight | (3456, 1152) | 7.9 MB |
| Q/K/V states | 3 × (392, 16, 72) | ~0.5 MB |
| Attention map | (16, 392, 392) | ~4.8 MB |
| Output proj weight | (1152, 1152) | 2.6 MB |

---

## MLP: `Qwen3VLVisionMLP`

**소스**: `modeling_qwen3_vl.py` L46–56

```python
# L46-56
class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size          # 1152
        self.intermediate_size = config.intermediate_size  # 4304
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]        # SiLU

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
```

### 파라미터

| 항목 | 값 |
|------|----|
| 입력 차원 | 1152 |
| 중간 차원 | 4304 |
| 출력 차원 | 1152 |
| 활성화 함수 | SiLU (Swish) |
| 확장 비율 | 4304 / 1152 ≈ 3.73 |
| fc1 파라미터 수 | 1152 × 4304 + 4304 = 4,954,112 + 4,304 |
| fc2 파라미터 수 | 4304 × 1152 + 1152 = 4,954,128 + 1,152 |

**주의**: VLM MLP (gate_proj + up_proj + down_proj, SwiGLU)와 달리, Vision MLP는 단순 2-layer FFN + SiLU 구조 (gate projection 없음).

---

## 27-Layer 순차 처리

**소스**: `modeling_qwen3_vl.py` L584, L738–751

```python
# L584
self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
# config.depth = 27

# L738-751 (forward)
for layer_num, blk in enumerate(self.blocks):
    hidden_states = blk(
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )
    # DeepStack: 특정 레이어에서 중간 특징 추출 (VLM early layer에 주입)
    if layer_num in self.deepstack_visual_indexes:
        deepstack_feature = self.deepstack_merger_list[...](hidden_states)
        deepstack_feature_lists.append(deepstack_feature)

hidden_states = self.merger(hidden_states)  # Stage 3으로 이동
```

### DeepStack 기능

- 지정 레이어(`deepstack_visual_indexes`)에서 중간 hidden_states를 `Qwen3VLVisionPatchMerger`로 투영
- VLM 초기 레이어에 직접 주입 → 시각 정보를 다층으로 활용
- 최종 레이어 출력은 별도로 `self.merger`를 통해 Stage 3으로 전달

---

## 출력

```
hidden_states: (N_patches, 1152)   — 27번째 레이어 출력
deepstack_feature_lists: list      — 중간 레이어 투영 특징 (VLM Deepstack 입력)
```

---

## 전체 파라미터 수 (Vision Transformer 27 Blocks)

| 컴포넌트 | 파라미터 수 (단일 레이어) | × 27 |
|----------|--------------------------|-------|
| QKV Linear (1152→3456) | 3,981,312 + 3,456 | 107.7 M |
| Out Linear (1152→1152) | 1,327,104 + 1,152 | 35.9 M |
| MLP fc1 (1152→4304) | 4,958,208 + 4,304 | 134.1 M |
| MLP fc2 (4304→1152) | 4,958,208 + 1,152 | 134.1 M |
| 2 × LayerNorm (1152) | 2,304 | 0.06 M |
| **합계** | **~15.2 M** | **~411.9 M** |

---

## 파이프라인 내 위치

```
Stage 1: Patch Embedding
  ↓ (N_patches, 1152)
Stage 2: Vision Transformer (27 Blocks) ← 현재 단계
  ↓ (N_patches, 1152)
Stage 3: Patch Merger
```

---

## 관련 파일

- 소스 코드: `modeling_qwen3_vl.py` (L46–56, L168–248, L251–275, L564–753)
  - 전체 경로: `alpamayo/ar1_venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- 코드 참조: [code_references/vision_encoder.md](code_references/vision_encoder.md)
- 시각화: [figures/vision_encoder_detail.png](figures/vision_encoder_detail.png)
