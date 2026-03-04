# Stage 6: Action → Trajectory 변환

## 개요

Unicycle kinematic model을 사용하여 (acceleration, curvature) action을 (x, y, yaw) trajectory로 변환.
수치 적분(사다리꼴 법칙)을 통해 64개 waypoint의 3D 위치 및 회전 행렬 계산.

**코드 위치**: `unicycle_accel_curvature.py` L300–382 (`action_to_traj`)

---

## 입력

```
action tensor: (B, 64, 2)   — Stage 5 Diffusion Sampling 출력
  [:, :, 0]: acceleration (정규화 상태)
  [:, :, 1]: curvature κ   (정규화 상태)
```

- `B`: batch size
- `64`: waypoint 수 (T=64 스텝, dt=0.1s → 총 6.4초 예측 구간)

---

## 변환 과정

### Step 1: 역정규화 (L319–326)

```python
accel = accel * accel_std + accel_mean     # [m/s²]
kappa = kappa * kappa_std + kappa_mean     # [1/m]  (곡률)
```

학습 시 정규화한 통계량(mean, std)으로 물리 단위 복원.

---

### Step 2: 초기 속도 추정 (L328–331)

```python
v0 = estimate_v0(history_trajectory)       # [m/s]
```

히스토리 trajectory에서 현재 속도 `v0` 추정.
이후 적분의 초기 조건으로 사용.

---

### Step 3: 속도 적분 (L335–341)

```python
velocity = [v0] + cumsum(accel × dt)      # shape: (N+1,)  N=64
```

- `v[0] = v0`
- `v[k+1] = v[k] + accel[k] × dt`
- 오일러 전진 적분으로 각 스텝 속도 계산
- 결과 shape: `(65,)` — 경계값 포함 (N+1 점)

---

### Step 4: 방향각 적분 (L343–353)

Unicycle 모델에서 방향각 변화율: `dθ/dt = κ × v`

가속도를 고려한 2차 보정 항 포함:

```python
dtheta = kappa * v[:-1] * dt + kappa * accel * dt² / 2

theta = [0] + cumsum(dtheta)              # shape: (N+1,)
```

- `θ[0] = 0` (현재 heading 기준 상대 좌표계)
- 가속도 보정항 `κ × accel × dt²/2`로 정확도 향상
- 결과 shape: `(65,)` — 경계값 포함

수식 전개:
```
θ_{k+1} = θ_k + κ_k × v_k × dt + κ_k × a_k × dt² / 2
```

---

### Step 5: 위치 적분 — 사다리꼴 법칙 (L354–366)

오일러 전진보다 정확한 사다리꼴 수치 적분:

```python
# x 방향 (전진)
dx = v[:-1] * cos(theta[:-1]) * dt/2 + v[1:] * cos(theta[1:]) * dt/2

# y 방향 (횡방향)
dy = v[:-1] * sin(theta[:-1]) * dt/2 + v[1:] * sin(theta[1:]) * dt/2

x = cumsum(dx)    # shape: (64,)
y = cumsum(dy)    # shape: (64,)
```

수식:
```
x_{k+1} = x_k + (v_k × cos θ_k + v_{k+1} × cos θ_{k+1}) × dt / 2
y_{k+1} = y_k + (v_k × sin θ_k + v_{k+1} × sin θ_{k+1}) × dt / 2
```

- 스텝 시작/끝 값 평균으로 곡선 경로를 더 정확하게 근사

---

### Step 6: 회전 행렬 변환 (L380)

```python
rot_2d = [[cos θ, -sin θ],
           [sin θ,  cos θ]]        # (64, 2, 2)

rot_3d = embed_in_3d(rot_2d)       # (64, 3, 3)
```

- 평면 yaw angle `θ`를 3D 회전 행렬로 확장
- z축 방향 회전 (roll=0, pitch=0)

---

## 출력

```
traj_future_xyz: (B, 64, 3)       — [x, y, z] 위치 (미터)
traj_future_rot: (B, 64, 3, 3)    — SO(3) 회전 행렬
```

- z 좌표: 평면 주행 가정 시 0 (또는 지면 높이)
- 좌표계: 현재 차량 위치/방향 기준 상대 좌표

---

## 타임라인

```
dt = 0.1s, N = 64 waypoints
총 예측 구간: 6.4초
```

| waypoint | 시간 | 예측 범위 |
|----------|------|----------|
| 0 | 0.1s | 근거리 |
| 31 | 3.2s | 중거리 |
| 63 | 6.4s | 원거리 |

---

## 관련 파일

- [`code_references/action_space.md`](code_references/action_space.md) — Unicycle action space 구현
- `unicycle_accel_curvature.py` L300–382
- [`figures/action_trajectory.png`](figures/action_trajectory.png)
