# Action Input Projection 코드 레퍼런스

> **원본 경로**: `alpamayo/src/alpamayo_r1/models/action_in_proj.py`

## 모듈 구성 개요

| 클래스 | 역할 | 입력 | 출력 |
|---|---|---|---|
| `RMSNorm` (L22-L35) | Root Mean Square 정규화 | `(..., dim)` | `(..., dim)` |
| `MLPEncoder` (L38-L70) | 다층 MLP 인코더 | `(B, C)` | `(B, outdim)` |
| `FourierEncoderV2` (L73-L101) | 로그 간격 Fourier 특징 인코더 | `(...,)` | `(..., dim)` |
| `PerWaypointActionInProjV2` (L104-L166) | 웨이포인트별 action 투영 | `(B, T, 2)` | `(B, T, out_dim)` |

---

## RMSNorm (L22-L35)

Root Mean Square 기반 정규화. LayerNorm 대비 평균 계산 생략.

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 학습 가능한 스케일 파라미터

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # RMS 정규화: x / sqrt(mean(x²) + eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)  # FP32로 계산 후 원래 dtype 복원
        return output * self.weight
```

---

## MLPEncoder (L38-L70)

기본 MLP 인코더. `num_enc_layers=4` 기준 구조:

```
Linear(num_input_feats → hidden_size) → SiLU
→ [RMSNorm → Linear(hidden_size → hidden_size) → SiLU] × 3
→ RMSNorm → Linear(hidden_size → outdim)
```

```python
class MLPEncoder(nn.Module):
    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        assert 1 <= num_enc_layers

        enc_layers = [
            nn.Linear(num_input_feats, hidden_size),
            nn.SiLU(),
        ]
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                enc_layers.extend([
                    RMSNorm(hidden_size, eps=1e-5),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                ])
            else:  # 마지막 레이어: outdim으로 출력, 활성화 없음
                enc_layers.extend([
                    RMSNorm(hidden_size, eps=1e-5),
                    nn.Linear(hidden_size, outdim),
                ])
        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C) -> (B, outdim)"""
        return self.trunk(x)
```

**`PerWaypointActionInProjV2` 기본값 기준 파라미터:**
- `num_input_feats=60`, `num_enc_layers=4`, `hidden_size=1024`, `outdim=2048`
- 총 레이어: `Linear(60→1024)` + `SiLU` + (`RMSNorm+Linear+SiLU`) × 3 + `RMSNorm+Linear(1024→2048)`

---

## FourierEncoderV2 (L73-L101)

로그 간격 주파수 기반 Fourier feature encoding. 스칼라 입력을 고차원 특징으로 변환.

```python
class FourierEncoderV2(nn.Module):
    def __init__(self, dim: int, max_freq: float = 100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        # [1.0, ..., 100.0] 로그 간격 — 저주파부터 고주파까지 균등 커버
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])  # (1, half) — 학습 불가 고정 버퍼

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: 임의 형상 (...,) 스칼라 입력
        arg = x[..., None] * self.freqs * 2 * torch.pi  # (..., half)
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)
        # 출력: (..., dim) — sin(half) || cos(half), sqrt(2) 정규화 인수
```

**핵심 포인트:**
- 입력: 스칼라 `(...,)` → 출력: `(..., dim)`
- 기본 설정 `dim=20`: sin 10차원 + cos 10차원
- 주파수 범위: 1~100Hz (로그 간격) — 다양한 스케일의 패턴 포착
- `sqrt(2)` 스케일링: 에너지 정규화 (각 주파수 기여도 균등화)
- `register_buffer`: 학습 대상 아님, 디바이스 이동 자동 추적

---

## PerWaypointActionInProjV2 (L104-L166)

각 웨이포인트별로 독립적으로 action을 Expert Decoder 입력 차원으로 투영.

```python
class PerWaypointActionInProjV2(torch.nn.Module):
    def __init__(
        self,
        in_dims: list[int],        # 예: (64, 2) — (n_waypoints, action_dim)
        out_dim: int,               # 예: 2048 — expert hidden_size
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim

        # action_dim개의 Fourier 인코더 (accel용, curvature용 각각 독립)
        sinus = []
        for _ in range(in_dims[-1]):  # in_dims[-1] = 2
            sinus.append(FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq))
        self.sinus = nn.ModuleList(sinus)  # 2개 FourierEncoderV2

        # timestep 전용 Fourier 인코더
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)

        # 총 입력 차원: 20(accel) + 20(kappa) + 20(timestep) = 60
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        # = 20 + 20 + 20 = 60

        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,  # 60
            num_enc_layers=num_enc_layers,    # 4
            hidden_size=hidden_size,          # 1024
            outdim=out_dim,                   # 2048
        )
        self.norm = nn.LayerNorm(out_dim)     # 최종 LayerNorm (RMSNorm 아님)
```

### forward (L148-L166)

```python
def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    # x: (B, T, 2)  — action sequence (B=batch, T=64 waypoints, 2=accel+kappa)
    # timesteps: (B, ...) — diffusion timestep
    B, T, _ = x.shape

    # 각 action 차원(accel, kappa)을 별도 Fourier encoding
    action_feats = torch.cat(
        [s(x[:, :, i]) for i, s in enumerate(self.sinus)],
        dim=-1
    )  # (B, T, 40) = (B, T, 20+20)

    # timestep Fourier encoding → 모든 웨이포인트에 동일 적용
    timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])  # (B, 1, 20)
    timestep_feats = timestep_feats.repeat(1, T, 1)                     # (B, T, 20)

    # 특징 결합
    x = torch.cat((action_feats, timestep_feats), dim=-1)  # (B, T, 60)

    # MLP 인코딩: 웨이포인트 차원을 배치로 flatten하여 병렬 처리
    return self.norm(
        self.encoder(x.flatten(0, 1)).reshape(B, T, -1)
    )  # (B*T, 60) → MLP → (B*T, 2048) → reshape → (B, T, 2048) → LayerNorm
```

**핵심 포인트:**
- 각 action 차원(accel, curvature)을 별도 `FourierEncoderV2`로 독립 인코딩 — 차원 간 주파수 간섭 방지
- timestep 인코딩은 단일 스칼라 추출(`timesteps[..., -1]`) 후 전 웨이포인트 broadcast
- `x.flatten(0, 1)`: `(B, T, 60)` → `(B*T, 60)` — 웨이포인트를 독립 샘플로 처리
- 최종 `nn.LayerNorm` (내부 레이어 `RMSNorm`과 구별): 출력 안정화
- 입력 흐름 요약: `(B,T,2)` → Fourier `(B,T,40)` + timestep `(B,T,20)` → MLP `(B,T,2048)` → LayerNorm

---

## 데이터 흐름 요약

```
action (B, T, 2)
    │
    ├─ accel  (B, T) ──→ FourierEncoderV2 ──→ (B, T, 20)
    └─ kappa  (B, T) ──→ FourierEncoderV2 ──→ (B, T, 20)
                                                   │
timesteps (B, ...) ──→ FourierEncoderV2 ──→ (B, 1, 20) ──→ repeat ──→ (B, T, 20)
                                                   │
                          cat ────────────────────────────────────→ (B, T, 60)
                           │
                         flatten(0,1) ──→ (B*T, 60)
                           │
                        MLPEncoder ──→ (B*T, 2048)
                           │
                         reshape ──→ (B, T, 2048)
                           │
                        LayerNorm ──→ (B, T, 2048)  [Expert Decoder 입력]
```
