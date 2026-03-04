# Stage 1: Patch Embedding

## 개요

이미지 텐서를 고정 크기 패치로 분할하여 1152차원 임베딩 벡터로 변환하는 단계.
3D Conv (temporal + spatial) 를 단일 연산으로 적용, 공간 및 시간 축을 동시에 다운샘플링.
이후 3D Rotary Positional Embedding을 더해 위치 정보를 주입.

---

## 입력

```
hidden_states: (B × T × H_patches × W_patches, 3, temporal_patch_size, patch_size, patch_size)
```

실제로 Qwen3VL 파이프라인에서는 이미지를 사전에 낱개 패치 단위로 펼쳐(unfolding) 전달:
- `C = 3` (RGB)
- `temporal_patch_size = 2` (연속 2 프레임을 하나의 3D 패치로 합산)
- `patch_size = 16` (공간 축 16×16 픽셀)

grid_thw: (N_images, 3) — 각 이미지의 (T, H_patches, W_patches)

---

## 핵심 연산: Conv3d Patch Projection

### 클래스: `Qwen3VLVisionPatchEmbed`
**소스**: `modeling_qwen3_vl.py` L59–76

```python
# L59-68
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size           # 16
        self.temporal_patch_size = config.temporal_patch_size  # 2
        self.in_channels = config.in_channels          # 3
        self.embed_dim = config.hidden_size            # 1152

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim,
            kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # L70-75
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
```

### 파라미터 상세

| 항목 | 값 |
|------|----|
| 연산 | `nn.Conv3d` |
| 입력 채널 (`in_channels`) | 3 |
| 출력 채널 (`embed_dim`) | 1152 |
| 커널 크기 | (2, 16, 16) |
| 스트라이드 | (2, 16, 16) — 커널과 동일 → 패치 간 중첩 없음 |
| Bias | True |
| 가중치 shape | `(1152, 3, 2, 16, 16)` |

### 출력 shape 계산

입력 이미지 해상도 `H × W`, `T` 프레임 가정 시:
```
N_patches = (T / 2) × (H / 16) × (W / 16)
출력: (N_patches, 1152)
```

예시 (224×224, 4 프레임):
```
N_patches = 2 × 14 × 14 = 392
출력: (392, 1152)
```

---

## Rotary Position Embedding (3D RoPE)

### 클래스: `Qwen3VLVisionRotaryEmbedding`
**소스**: `modeling_qwen3_vl.py` L79–90

```python
# L79-90
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
```

### 적용 방식: `Qwen3VLVisionModel.rot_pos_emb`
**소스**: `modeling_qwen3_vl.py` L603–640

- `head_dim = 1152 / 16 = 72`
- `rotary_dim = head_dim // 2 = 36`
- 각 패치의 (height, width) 2D 좌표를 freq_table에서 룩업하여 RoPE 벡터 생성
- `embeddings = freq_table[pos_ids]` → shape `(N_patches, 2, 36)` → flatten → `(N_patches, 72)`

```python
# L582
head_dim = config.hidden_size // config.num_heads  # 1152 // 16 = 72
self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)  # dim=36
```

### 적용: `apply_rotary_pos_emb_vision`
**소스**: `modeling_qwen3_vl.py` L116–127

```python
def apply_rotary_pos_emb_vision(q, k, cos, sin):
    # cos, sin shape: (seq_len, 1, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### 위치 임베딩 추가 (Absolute PE)

`Qwen3VLVisionModel.forward` L714–717에서 패치 임베딩에 절대 위치 임베딩 합산:

```python
# L714-717
hidden_states = self.patch_embed(hidden_states)
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
hidden_states = hidden_states + pos_embeds   # 절대 pos embed 추가
rotary_pos_emb = self.rot_pos_emb(grid_thw)  # RoPE는 Attention 내부에서 별도 적용
```

---

## 출력

```
hidden_states: (N_patches, 1152)
rotary_pos_emb: (N_patches, head_dim) — Attention 블록 내부 Q/K에 적용
```

---

## 파이프라인 내 위치

```
[원본 이미지] → Stage 1: Patch Embedding → [패치 임베딩 시퀀스]
                                           ↓
                              Stage 2: Vision Transformer (27 Blocks)
```

---

## 관련 파일

- 소스 코드: `modeling_qwen3_vl.py` (L59–76, L79–90, L603–640, L703–720)
  - 전체 경로: `alpamayo/ar1_venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
- 코드 참조: [code_references/vision_encoder.md](code_references/vision_encoder.md)
- 시각화: [figures/vision_encoder_detail.png](figures/vision_encoder_detail.png)
