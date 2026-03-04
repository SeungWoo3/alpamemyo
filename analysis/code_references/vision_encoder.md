# Vision Encoder 코드 레퍼런스

> **원본 경로**: `/home/seungwoo/workspace/alpamayo/ar1_venv/lib/python3.12/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
>
> **모델**: Qwen3-VL Vision Encoder (NVlabs Alpamayo-R1-10B 기반)
> **비고**: `modular_qwen3_vl.py`에서 자동 생성된 파일 — 직접 수정 불가

---

## 1. PatchEmbed (L59–L76)

Conv3d를 사용하여 입력 이미지/비디오 픽셀을 1152차원 패치 벡터로 변환. 공간(16×16) 및 시간(2프레임) 차원을 동시에 압축.

```python
class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size            # 16
        self.temporal_patch_size = config.temporal_patch_size  # 2
        self.in_channels = config.in_channels          # 3 (RGB)
        self.embed_dim = config.hidden_size            # 1152

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim,
            kernel_size=kernel_size, stride=kernel_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states
```

**핵심 포인트:**
- `Conv3d(3, 1152, kernel=(2,16,16), stride=(2,16,16))` — 시공간 패치 동시 처리
- 입력: `(N, C, T, H, W)` 형태로 reshape 후 Conv3d 적용
- 출력: `(num_patches, 1152)` — 각 패치가 1152차원 벡터로 표현
- `temporal_patch_size=2`이므로 연속 2프레임이 하나의 시간 토큰으로 합산

---

## 2. Rotary Embedding (L79–L90)

Vision Transformer용 2D RoPE 주파수 테이블 생성. 시퀀스 길이에 대한 사인/코사인 기저 주파수 계산.

```python
class Qwen3VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
```

**핵심 포인트:**
- `dim = head_dim // 2 = 72 // 2 = 36` — 행/열 방향 각각 36차원씩 할당
- `inv_freq`: 고정 버퍼 (학습 안 됨), `theta=10000` 기본값
- `forward(seqlen)` → `(seqlen, 36)` 주파수 테이블 반환
- `VisionModel.rot_pos_emb()`에서 H/W 좌표 2개를 각각 lookup하여 `(N_tokens, 72)` 구성 후 concat → `(N_tokens, 144)` 최종 RoPE

---

## 3. Vision Block (L251–L275)

단일 Transformer 블록. Pre-norm 구조: LayerNorm → Attention → 잔차 연결 → LayerNorm → MLP → 잔차 연결. 총 27개 블록이 `VisionModel`에서 순차 적용됨.

```python
class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
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

**Attention 레이어 (`Qwen3VLVisionAttention`, L168–L248):**

```python
class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size        # 1152
        self.num_heads = config.num_heads    # 16
        self.head_dim = self.dim // self.num_heads  # 72
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.is_causal = False

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Flash Attention 2: cu_seqlens로 가변 길이 배치 처리
        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self, query_states, key_states, value_states,
                attention_mask=None, scaling=self.scaling,
                cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen, max_length_k=max_seqlen,
                is_causal=False, **kwargs,
            )
        # 기타 구현 (sdpa, eager): 이미지별 분할 후 개별 처리
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(t, lengths.tolist(), dim=2)
                      for t in (query_states, key_states, value_states)]
            attn_outputs = [attention_interface(self, q, k, v, ...)[0]
                            for q, k, v in zip(*splits)]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)
```

**핵심 포인트:**
- Attention: `16 heads`, `head_dim=72`, `hidden=1152` — 이미지/비디오 배치를 packed sequence로 처리
- MLP (`Qwen3VLVisionMLP`): `1152 → 4304 → 1152`, 활성화 함수 `quick_gelu`
- `cu_seqlens`: 배치 내 각 이미지 경계를 표시하는 누적 시퀀스 길이 (packed attention 핵심)
- `is_causal=False` — Vision 인코더는 양방향 어텐션 (마스크 없음)
- Flash Attention 2 사용 시 `cu_seqlens`를 직접 FA2에 전달 → padding 없이 효율적 배치 처리
- `GradientCheckpointingLayer` 상속 — 메모리 절약을 위한 선택적 gradient checkpointing 지원

---

## 4. Patch Merger (L93–L106)

`spatial_merge_size=2`이므로 인접 2×2 패치 4개를 하나로 병합. `hidden_size × 4 → out_hidden_size`로 차원 변환하여 LLM 입력 형식에 맞춤.

```python
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)  # 1152 * 4 = 4608
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)   # 4608 → 4608
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)  # 4608 → 3584

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x
```

**핵심 포인트:**
- 입력: `(N_patches, 1152)` → reshape: `(N_patches/4, 4608)` — 공간 2×2 병합
- 출력: `(N_patches/4, 3584)` — LLM hidden size(`out_hidden_size=3584`)에 맞춤
- `use_postshuffle_norm=False` (기본 merger): reshape 후 LayerNorm 적용
- `use_postshuffle_norm=True` (deepstack merger): 개별 패치 단계에서 LayerNorm 먼저 적용
- **DeepStack**: 블록 8, 16, 24에서 중간 피처를 별도 merger로 추출 → 텍스트 디코더 초기 레이어에 주입

---

## 5. Vision Model (L564–L753)

전체 Vision 인코더 파이프라인 조율. PatchEmbed → Positional Embed → 27× VisionBlock → PatchMerger 순서로 실행. DeepStack 중간 피처 추출 포함.

```python
class Qwen3VLVisionModel(Qwen3VLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size  # 2
        self.patch_size = config.patch_size                  # 16

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        # num_position_embeddings=2304 (48×48 그리드), 학습 가능한 절대 위치 임베딩

        head_dim = config.hidden_size // config.num_heads  # 72
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)  # dim=36

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])  # 27개
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

        # DeepStack: 블록 [8, 16, 24]에서 중간 피처 추출용 별도 merger
        self.deepstack_visual_indexes = config.deepstack_visual_indexes  # [8, 16, 24]
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
            for _ in range(len(config.deepstack_visual_indexes))
        ])

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
        # 1. 패치 임베딩
        hidden_states = self.patch_embed(hidden_states)

        # 2. 절대 위치 임베딩 (이중선형 보간으로 임의 해상도 지원)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 3. Rotary 위치 임베딩 준비 (2D RoPE: 행/열 좌표 각각)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # 4. 패치별 시퀀스 경계 (packed attention용 cu_seqlens)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # 5. 27개 블록 순차 실행 + DeepStack 중간 피처 추출
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings, **kwargs)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        # 6. 최종 PatchMerger (2×2 공간 병합 → LLM 입력 차원 변환)
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists
```

**핵심 포인트:**
- 전체 파이프라인: `pixel_values → (N_patches, 1152) → 27×Block → (N_patches, 1152) → Merger → (N_patches/4, 3584)`
- `grid_thw`: `(num_images, 3)` 텐서 — 각 이미지의 `(T, H, W)` 그리드 크기
- `fast_pos_embed_interpolate()`: 학습된 48×48 절대 위치 임베딩을 임의 해상도로 이중선형 보간
- `rot_pos_emb()`: 행(H)·열(W) 좌표 각각에 대한 RoPE 주파수 lookup → 2D 공간 인식
- **DeepStack 통합**: 블록 8/16/24의 중간 hidden state를 추출해 별도 merger 적용 후 LLM 초기 레이어에 주입 → 시각 정보를 LLM 전체 깊이에 분산
- 최종 출력 `(N_patches/4, 3584)`이 LLM 텍스트 토큰과 동일 차원으로 병합됨
