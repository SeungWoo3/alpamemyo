# Stage 5: Diffusion Sampling (Flow Matching)

## 개요

Flow Matching 기법으로 10 Euler 스텝을 통해 노이즈에서 action trajectory 생성.
각 스텝마다 `step_fn`이 호출되며, 내부에서 action_in_proj → Expert Decoder → action_out_proj 순으로 속도장(velocity field) 예측.
10번의 적분 후 최종 denoised action tensor 출력.

**코드 위치**: `flow_matching.py` L89–127, `alpamayo_r1.py` L254–297

---

## Flow Matching 원리

### 학습 시 (Forward Process)
```
x_t = (1 - t) * x_1 + t * x_0

x_1 = 데이터 (실제 action)
x_0 = 노이즈 ~ N(0, 1)
t ∈ [0, 1]
```

목표 속도장: `v* = x_1 - x_0`

네트워크는 임의의 `(x_t, t)` 쌍에서 `v*`를 예측하도록 학습.

### 추론 시 (Reverse Process)
```
t: 0 → 1  (노이즈에서 데이터 방향)
x_{t+dt} = x_t + dt * v_θ(x_t, t)
```

- **주의**: `t=0`이 순수 노이즈, `t=1`이 데이터. Diffusion과 방향 반대.
- `flow_matching.py` L112: `linspace(0, 1, 11)` → 0.0, 0.1, ..., 1.0

---

## 입력

### 초기 노이즈
```python
x = randn(batch_size, *x_dims)   # flow_matching.py L111
# x shape: (B, 64, 2)
```
- `64`: waypoint 수
- `2`: action 차원 (acceleration, curvature)
- 표준 정규분포에서 샘플링

---

## 10 Euler Steps

**위치**: `flow_matching.py` L117–122

```python
time_steps = linspace(0, 1, 11)   # [0.0, 0.1, ..., 1.0]

for i in range(10):
    dt = time_steps[i+1] - time_steps[i]   # = 0.1
    t_start = time_steps[i]
    v = step_fn(x, t_start)                # velocity field 예측
    x = x + dt * v                         # Euler 적분
```

### 각 스텝의 텐서 흐름
| 단계 | 입력 shape | 출력 shape |
|------|-----------|-----------|
| 초기 | `(B, 64, 2)` 노이즈 | — |
| 스텝 k (k=0..9) | `x_k: (B, 64, 2)`, `t_k` | `x_{k+1}: (B, 64, 2)` |
| 최종 | — | `x_10: (B, 64, 2)` denoised |

---

## step_fn 내부 구조

**위치**: `alpamayo_r1.py` L255–284 (closure)

### 1. Action In-Projection
**파일**: `action_in_proj.py` L148–166 (`PerWaypointActionInProjV2.forward`)

```
입력: noisy action (B, 64, 2) + timestep scalar t
  ↓
각 action 차원별 Fourier 인코딩 (독립적으로)
  + timestep Fourier 인코딩
  ↓
Concat → shape: (B, 64, 60)
  ↓
MLP: 60 → 1024 → ... → 2048
  ↓
LayerNorm
  ↓
출력: (B, 64, 2048)
```

- 각 waypoint를 독립적으로 처리 (`PerWaypoint`)
- Fourier feature로 고주파 정보 보존
- MLP로 Expert Decoder 입력 차원(2048)에 맞춤

### 2. Expert Decoder
```
입력: (B, 64, 2048)  +  KV cache (Stage 4 VLM 출력)
  ↓
16 Transformer layers
  - Self-attention: (B, 64, 2048)
  - Cross-attention: KV cache 참조 (VLM context 조건부)
  ↓
출력: (B, 64, 2048)
```

- VLM의 reasoning context를 cross-attention으로 참조
- action 시퀀스 전체에 걸쳐 일관성 있는 속도장 예측

### 3. Action Out-Projection
```
입력: (B, 64, 2048)
  ↓
Linear: 2048 → 2
  ↓
출력: (B, 64, 2)   # 예측 속도장 v_θ
```

---

## 전체 step_fn 흐름 요약

```
(B, 64, 2) + t
      ↓
  action_in_proj          (B, 64, 2048)
      ↓
  Expert Decoder           (B, 64, 2048)
      ↓
  action_out_proj          (B, 64, 2)  ← v_θ(x, t)
      ↓
  x = x + 0.1 * v
```

---

## 출력

```
denoised action tensor: (B, 64, 2)
```
- `[:,  :, 0]`: acceleration (정규화 상태)
- `[:, :, 1]`: curvature (정규화 상태)
- Stage 6 action_to_traj 변환으로 전달

---

## 관련 파일

- [`code_references/diffusion.md`](code_references/diffusion.md) — Flow Matching 구현
- [`code_references/action_in_proj.md`](code_references/action_in_proj.md) — Action In-Projection
- `flow_matching.py` L89–127
- `alpamayo_r1.py` L254–297
- [`figures/diffusion_steps.png`](figures/diffusion_steps.png)
