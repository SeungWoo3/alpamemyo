# Diffusion (Flow Matching) 코드 레퍼런스

> **원본 경로**: `alpamayo/src/alpamayo_r1/diffusion/flow_matching.py`

## FlowMatching 클래스 (L22-L127)

Flow Matching 기반 생성 모델. Euler integration으로 가우시안 노이즈에서 denoised action을 샘플링.
`BaseDiffusion` (`diffusion/base.py`) 상속.

---

### 초기화 (L32-L47)

```python
class FlowMatching(BaseDiffusion):
    def __init__(
        self,
        int_method: Literal["euler"] = "euler",
        num_inference_steps: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)  # x_dims를 BaseDiffusion에 전달
        self.int_method = int_method
        self.num_inference_steps = num_inference_steps
```

**핵심 포인트:**
- `int_method`: 현재 `"euler"` 단일 지원
- `num_inference_steps`: 기본값 10 (Euler step 횟수)
- `x_dims`: `BaseDiffusion.__init__`에서 수신, `self.x_dims = list(x_dims)` 로 저장 (`base.py` L60)
  - AlpamayoR1에서 `get_action_space_dims()` 반환값 전달 → `[64, 2]` (64 waypoints × 2 action dims)

---

### sample 메서드 (L49-L87)

```python
@torch.no_grad()
def sample(
    self,
    batch_size: int,
    step_fn: StepFn,
    device: torch.device = torch.device("cpu"),
    return_all_steps: bool = False,
    inference_step: int | None = None,
    int_method: Literal["euler"] | None = None,
    *args,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # 호출 시 오버라이드 가능, 없으면 초기화 값 사용
    int_method = int_method or self.int_method
    inference_step = inference_step or self.num_inference_steps
    if int_method == "euler":
        return self._euler(
            batch_size=batch_size,
            step_fn=step_fn,
            device=device,
            return_all_steps=return_all_steps,
            inference_step=inference_step,
        )
    else:
        raise ValueError(f"Invalid integration method: {int_method}")
```

**핵심 포인트:**
- `@torch.no_grad()`: 추론 전용, gradient 계산 비활성화
- `inference_step` / `int_method` 인자로 호출 시 오버라이드 지원
- `StepFn`: `base.py` L26-L42에 정의된 Protocol, `(*, x: Tensor, t: Tensor) -> Tensor` 시그니처
- `return_all_steps=True` 시 `(all_steps_tensor [B, T+1, *x_dims], time_steps [T+1])` 반환

---

### _euler — 핵심 추론 루프 (L89-L127)

```python
def _euler(
    self,
    batch_size: int,
    step_fn: StepFn,
    device: torch.device = torch.device("cpu"),
    return_all_steps: bool = False,
    inference_step: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # 가우시안 노이즈에서 시작 (t=0 상태)
    x = torch.randn(batch_size, *self.x_dims, device=device)  # (B, 64, 2)

    # t: 0.0 → 1.0, inference_step+1 개의 균등 간격 시간점
    time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
    n_dim = len(self.x_dims)
    if return_all_steps:
        all_steps = [x]

    for i in range(inference_step):  # 10회 반복 (기본값)
        # dt 및 t_start를 x와 broadcast 가능한 형태로 reshape
        dt = time_steps[i + 1] - time_steps[i]                          # 스칼라: 0.1
        dt = dt.view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)  # (B, 1, 1)
        t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch_size, *[1] * n_dim)  # (B, 1, 1)

        v = step_fn(x=x, t=t_start)   # Expert Decoder가 예측한 velocity field (B, 64, 2)
        x = x + dt * v                 # Euler step: x_{i+1} = x_i + dt * v

        if return_all_steps:
            all_steps.append(x)

    if return_all_steps:
        return torch.stack(all_steps, dim=1), time_steps  # ([B, T+1, 64, 2], [T+1])
    return x  # (B, 64, 2) — denoised action
```

**핵심 포인트:**
- t 방향: 0 → 1 (Flow Matching 관례: 0 = 순수 가우시안 노이즈, 1 = 실제 데이터 분포)
- `dt = 0.1` (10 step 균등 분할), `t_start = 0.0, 0.1, ..., 0.9`
- `x_dims = [64, 2]`: 64 waypoints × 2 action 차원 (accel, curvature) — unicycle 모델
- `step_fn`: `AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout` 내 정의된 클로저
  - 내부 흐름: `action_in_proj(x, t)` → Expert Decoder (VLM KV cache 조건부) → `action_out_proj`
- `dt.view` / `t_start.view`: n_dim=2 기준으로 `(B, 1, 1)` 형태로 브로드캐스트 준비
- `prompt_cache.crop(prefill_seq_len)`: `step_fn` 내부에서 각 Euler step마다 KV cache 복원

---

## BaseDiffusion

> **원본 경로**: `alpamayo/src/alpamayo_r1/diffusion/base.py` L45-L88

```python
class BaseDiffusion(ABC, nn.Module):
    def __init__(
        self,
        x_dims: list[int] | tuple[int] | int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.x_dims = [x_dims] if isinstance(x_dims, int) else list(x_dims)

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        step_fn: StepFn,
        device: torch.device = torch.device("cpu"),
        return_all_steps: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
```

**핵심 포인트:**
- `x_dims`: int 단일값 입력 시 리스트로 감싸 통일 처리
- `sample` 추상 메서드: `FlowMatching`에서 구현

---

## StepFn Protocol

> **원본 경로**: `alpamayo/src/alpamayo_r1/diffusion/base.py` L26-L42

```python
class StepFn(Protocol):
    def __call__(
        self,
        *,
        x: torch.Tensor,  # noisy action, (B, *x_dims)
        t: torch.Tensor,  # timestep, broadcast 가능
    ) -> torch.Tensor:    # velocity field, (B, *x_dims)
        ...
```

**핵심 포인트:**
- `Protocol` 타입: 구조적 서브타이핑, 명시적 상속 없이 시그니처만 일치하면 호환
- keyword-only 인자 (`*` 구분자): `step_fn(x=x, t=t_start)` 형태로만 호출 가능
- `FlowMatching._euler`에서 `v = step_fn(x=x, t=t_start)` 형태로 호출

---

## PerWaypointActionInProjV2 — action_in_proj 구현체

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/action_in_proj.py` L104-L166

```python
class PerWaypointActionInProjV2(torch.nn.Module):
    def __init__(
        self,
        in_dims: list[int],      # [64, 2] (waypoints, action_dim)
        out_dim: int,            # expert_config.hidden_size
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        # action_dim 수(=2)만큼 개별 Fourier 인코더 생성
        sinus = []
        for _ in range(in_dims[-1]):
            sinus.append(FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq))
        self.sinus = nn.ModuleList(sinus)
        # timestep 전용 Fourier 인코더
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """(B, 64, 2), timesteps (B, 1, 1) → (B, 64, out_dim)"""
        B, T, _ = x.shape

        # 각 action 차원(accel, curvature)별 Fourier 인코딩 후 concat
        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        # timestep Fourier 인코딩 후 T번 반복
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, T, 1)
        # action + timestep concat → MLP → LayerNorm
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
```

**핵심 포인트:**
- `FourierEncoderV2` (L73-L101): 로그 스케일 주파수 Fourier 인코딩, `sin/cos` concat → `num_fourier_feats` 차원 출력
- action dim(2)별 독립적인 Fourier 인코더: accel/curvature의 서로 다른 스케일 특성 반영
- `x.flatten(0, 1)`: `(B*T, input_feats)` 형태로 MLP 일괄 처리 후 `(B, T, out_dim)` 복원
- `timesteps[..., -1]`: `(B, 1, 1)` → 마지막 차원 추출 → Fourier 인코딩 후 `(B, T, feats)` 브로드캐스트
