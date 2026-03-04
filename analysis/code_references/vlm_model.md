# VLM 모델 코드 레퍼런스

## AlpamayoR1 클래스

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/alpamayo_r1.py`

### 1. 초기화 (L79-L120)

Expert Decoder, Diffusion, Action Space 모듈 구성.

```python
class AlpamayoR1(ReasoningVLA):
    def __init__(
        self,
        config: AlpamayoR1Config,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
    ):
        super().__init__(config, pretrained_modules, original_vocab_size, print_param_count=False)

        # VLM의 text_config를 복제하여 Expert 전용 설정 구성
        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)
        # Expert는 embed_tokens 불필요 (action_in_proj에서 직접 임베딩 수신)
        del self.expert.embed_tokens

        self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
        self.diffusion: BaseDiffusion = hyu.instantiate(
            config.diffusion_cfg,
            x_dims=self.action_space.get_action_space_dims(),
        )

        self.action_in_proj = hyu.instantiate(
            config.action_in_proj_cfg,
            in_dims=self.action_space.get_action_space_dims(),
            out_dim=expert_config.hidden_size,
        )
        self.action_out_proj = hyu.instantiate(
            config.action_out_proj_cfg,
            in_features=expert_config.hidden_size,
            out_features=self.action_space.get_action_space_dims()[-1],
        )

        # action 관련 모듈을 expert와 동일한 dtype으로 변환
        expert_dtype = self.expert.dtype
        if self.config.keep_same_dtype:
            self.diffusion = self.diffusion.to(dtype=expert_dtype)
            self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
            self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)

        self.post_init()
```

**핵심 포인트:**
- `expert`: VLM의 `text_config`를 복제하여 별도 디코더 생성 (hidden_size, num_layers 등 오버라이드 가능)
- `embed_tokens` 삭제: Expert는 `action_in_proj`에서 직접 임베딩 수신
- `action_in_proj`: noisy action `(B, 64, 2)` + timestep → Fourier 인코딩 → MLP → `(B, 64, hidden_size)`
- `action_out_proj`: Expert 출력 `(B, 64, hidden_size)` → velocity field `(B, 64, 2)`
- `keep_same_dtype` 플래그로 action 모듈의 dtype을 Expert와 일치시킴

---

### 2. 추론 메인 루프 — sample_trajectories_from_data_with_vlm_rollout (L122-L328)

#### 2a. VLM Autoregressive Generation (L162-L207)

```python
# 히스토리 trajectory를 input_ids에 융합
input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)

# Generation 설정
generation_config = self.vlm.generation_config
generation_config.top_p = top_p           # 0.98
generation_config.temperature = temperature  # 0.6
generation_config.do_sample = True
generation_config.num_return_sequences = num_traj_samples
generation_config.max_new_tokens = max_generation_length
generation_config.output_logits = True
generation_config.return_dict_in_generate = True

# <traj_future_start> 토큰 감지 후 중단하는 커스텀 stopping criteria
eos_token_id = self.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
stopping_criteria = StoppingCriteriaList([StopAfterEOS(eos_token_id=eos_token_id)])

# trajectory discrete 토큰의 logit을 -inf로 마스킹
logits_processor = LogitsProcessorList(
    [
        ExpertLogitsProcessor(
            traj_token_offset=self.config.traj_token_start_idx,
            traj_vocab_size=self.config.traj_vocab_size,
        )
    ]
)
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,
    logits_processor=logits_processor,
    **tokenized_data,
)
vlm_outputs.rope_deltas = self.vlm.model.rope_deltas

# EOS 이후 padding 처리
vlm_outputs.sequences = replace_padding_after_eos(
    token_ids=vlm_outputs.sequences,
    eos_token_id=eos_token_id,
    pad_token_id=self.tokenizer.pad_token_id,
)
prompt_cache = vlm_outputs.past_key_values
prefill_seq_len = prompt_cache.get_seq_length()
```

**핵심 포인트:**
- `fuse_traj_tokens`: 히스토리 trajectory를 discrete 토큰으로 인코딩하여 `input_ids`의 `<traj_history>` 플레이스홀더 교체
- `ExpertLogitsProcessor`: trajectory 토큰 범위(`traj_token_start_idx` ~ `+traj_vocab_size`)를 `-inf`로 마스킹, CoT 생성 품질 향상
- `StopAfterEOS`: `<traj_future_start>` 토큰 감지 시점에 생성 중단 → KV cache 확보
- `prompt_cache`: VLM이 생성한 KV cache를 Expert Decoder가 재사용

#### 2b. Position ID 및 Attention Mask 조정 (L232-L248)

```python
# diffusion 토큰 수 (= action waypoint 수)
n_diffusion_tokens = self.action_space.get_action_space_dims()[0]

# future 토큰용 position_ids 계산 (RoPE delta 반영)
position_ids = torch.arange(n_diffusion_tokens, device=device)
position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
delta = vlm_outputs.rope_deltas + offset[:, None]
position_ids += delta.to(position_ids.device)

# padding 토큰 제거를 위한 attention_mask 설정
attention_mask = torch.zeros(
    (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
    dtype=torch.float32,
    device=device,
)
for i in range(b_star):
    attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(
        attention_mask.dtype
    ).min
```

**핵심 포인트:**
- `offset`: 시퀀스별 `<traj_future_start>` 위치 기반 계산, VLM 생성 길이 차이 흡수
- `position_ids`: 3채널 (Qwen3-VL의 mrope 형식), VLM rope_delta 반영하여 연속성 보장
- `attention_mask`: padding 구간을 `-inf`로 마스킹, causal mask와 함께 적용

#### 2c. step_fn 클로저 — Denoising Step (L255-L284)

```python
def step_fn(
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    # x: (B*, *action_dim), t: timestep (broadcast 가능)
    b_star = x.shape[0]

    # noisy action + timestep → Expert 입력 임베딩
    future_token_embeds = self.action_in_proj(x, t)  # (B*, 64, hidden_size)
    if future_token_embeds.dim() == 2:
        future_token_embeds = future_token_embeds.view(b_star, n_diffusion_tokens, -1)

    # VLM KV cache를 조건으로 Expert Decoder 실행
    expert_out_base = self.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prompt_cache,   # VLM KV cache 재사용 (조건부 생성)
        attention_mask=attention_mask,
        use_cache=True,
        **forward_kwargs,
    )
    # 신규 추가된 토큰을 KV cache에서 제거 (재사용을 위해 prefill 길이로 복원)
    prompt_cache.crop(prefill_seq_len)

    last_hidden = expert_out_base.last_hidden_state  # (B*, Tf, hidden_size)
    last_hidden = last_hidden[:, -n_diffusion_tokens:]

    # Expert 출력 → velocity field 예측
    pred = self.action_out_proj(last_hidden).view(
        -1, *self.action_space.get_action_space_dims()
    )  # (B*, 64, 2)
    return pred
```

**핵심 포인트:**
- `action_in_proj` (`PerWaypointActionInProjV2`): 각 waypoint별 Fourier 인코딩 + timestep 인코딩 → MLP → LayerNorm
- `prompt_cache.crop(prefill_seq_len)`: Expert 포워드 후 KV cache를 prefill 길이로 복원, 다음 diffusion step에서 재사용 가능하게 함
- `expert_non_causal_attention` 플래그: non-causal attention 옵션 지원 (`is_causal=False`)
- `action_out_proj` (Linear): Expert hidden state → 2차원 velocity field (accel, curvature)

#### 2d. Diffusion Sampling (L287-L297)

```python
total_batch = B * n_samples_total  # B * (num_traj_samples * num_traj_sets)
sampled_action = self.diffusion.sample(
    batch_size=total_batch,
    step_fn=step_fn,
    device=device,
    return_all_steps=False,
    **diffusion_kwargs,
)
```

#### 2e. Action → Trajectory 변환 (L307-L308)

```python
pred_xyz, pred_rot = self.action_space.action_to_traj(
    sampled_action, hist_xyz_rep, hist_rot_rep
)
```

**핵심 포인트:**
- `sampled_action`: `(B * num_traj_samples, 64, 2)` — unicycle accel/curvature 표현
- `action_to_traj`: unicycle kinematic 모델로 accel/curvature → xyz 좌표 + rotation 변환
- `hist_xyz_rep`, `hist_rot_rep`: 히스토리 마지막 프레임을 `num_traj_samples` 배로 반복한 초기 조건

---

## fuse_traj_tokens

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/base_model.py` L169-L198

```python
def fuse_traj_tokens(
    self, input_ids: torch.Tensor, traj_data: dict[str, Any] | None = None
) -> torch.Tensor:
    """input_ids에 trajectory 토큰 융합."""
    if (
        traj_data is None
        or traj_data.get("ego_history_xyz") is None
        or traj_data.get("ego_history_rot") is None
    ):
        return input_ids

    has_future = "ego_future_xyz" in traj_data and traj_data["ego_future_xyz"] is not None
    attrs = self._validate_mixin_requirements(require_future=has_future)

    # 히스토리 trajectory를 discrete 토큰 인덱스로 인코딩
    hist_idx = tokenize_history_trajectory(
        attrs["hist_traj_tokenizer"], traj_data, attrs["hist_token_start_idx"]
    )
    # input_ids의 <traj_history> placeholder 토큰을 실제 인덱스로 교체
    input_ids = replace_pad_token(
        input_ids, hist_idx, attrs["config"].traj_token_ids["history"]
    )

    return input_ids
```

**핵심 포인트:**
- `TrajectoryFusionMixin`의 메서드, `ReasoningVLA`에서 믹스인으로 상속
- `tokenize_history_trajectory` (L92-L123): hist_xyz/rot → 토크나이저 encode → `hist_token_start_idx` 오프셋 적용
- `replace_pad_token` (L85-L89): `masked_scatter`로 `<traj_history>` 플레이스홀더를 실제 토큰 ID로 교체
- 추론 시(`traj_data_vlm`)에는 history만 융합, 학습 시에는 future까지 융합

---

## tokenize_history_trajectory

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/base_model.py` L92-L123

```python
def tokenize_history_trajectory(
    tokenizer: Any, traj_data: dict[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """히스토리 trajectory 토크나이즈. 입력 형상: (B, n_traj, T, 3).

    Returns:
        torch.Tensor: [B, n_traj * tokens_per_history_traj]
    """
    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,   # 히스토리 인코딩 시 hist_xyz를 fut_xyz로 전달
            fut_rot=hist_rot,
        )
        + start_idx
    )  # [B*n_traj, tokens_per_history_traj]
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)

    return hist_idx
```

**핵심 포인트:**
- `hist_xyz`를 `fut_xyz`에 전달: 토크나이저 인터페이스가 future 기준으로 설계되어 있어 히스토리를 future 위치에 전달
- `start_idx` 오프셋: trajectory vocab이 전체 vocab의 특정 오프셋부터 시작하는 구조 반영
- `einops.rearrange`: `(B * n_traj, tokens)` → `(B, n_traj * tokens)` 재배열

---

## VLM 초기화 (_initialize_qwenvl3_vlm)

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/base_model.py` L368-L382

```python
def _initialize_qwenvl3_vlm(self, config: ReasoningVLAConfig) -> None:
    """Qwen3-VL VLM 백본 초기화."""
    vlm_config = Qwen3VLConfig.from_pretrained(
        config.vlm_name_or_path,
        dtype=config.model_dtype,
        attn_implementation=config.attn_implementation,
    )
    self.original_vocab_size = vlm_config.text_config.vocab_size
    # trajectory 토큰 포함하여 vocab 확장
    vlm_config.text_config.vocab_size = config.vocab_size
    vlm_config.vocab_size = config.vocab_size
    self.vlm = Qwen3VLForConditionalGeneration(vlm_config)
```

**핵심 포인트:**
- Qwen3-VL-8B 기반 (`Qwen/Qwen3-VL-8B-Instruct`), Vision Encoder + LLM 통합 모델
- `original_vocab_size` 저장: 원본 임베딩 가중치 복원 등에 활용
- `config.vocab_size`: 원본 vocab + trajectory discrete 토큰 768개 + 특수 토큰 추가분
- `text_config.vocab_size`와 최상위 `vocab_size` 양쪽 모두 갱신 필요 (Qwen3VL 구조상)

---

## ExpertLogitsProcessor

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/alpamayo_r1.py` L41-L70

```python
class ExpertLogitsProcessor(LogitsProcessor):
    """discrete trajectory 토큰 logit 마스킹."""

    def __init__(self, traj_token_offset: int, traj_vocab_size: int):
        super().__init__()
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # trajectory 토큰 범위를 -inf로 마스킹 → VLM이 trajectory 토큰을 생성하지 않도록 방지
        scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float('-inf')
        return scores
```

**핵심 포인트:**
- VLM autoregressive 생성 중 trajectory discrete 토큰(`<i0>` ~ `<i767>`) 선택 차단
- CoT(Chain-of-Thought) 텍스트 생성 품질 향상 목적
- `traj_token_start_idx`부터 `traj_vocab_size`(768)개 범위 마스킹
