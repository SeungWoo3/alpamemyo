# Action Space 코드 레퍼런스

> **원본 경로**: `alpamayo/src/alpamayo_r1/action_space/unicycle_accel_curvature.py`

## UnicycleAccelCurvatureActionSpace 클래스 (L36-L382)

Unicycle kinematic model 기반 action space. 가속도(accel)와 곡률(curvature)을 제어 입력으로 사용.

---

### 초기화 및 설정 (L36-L100)

```python
class UnicycleAccelCurvatureActionSpace(ActionSpace):
    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        # Tikhonov 정규화 파라미터 (theta, v, a, kappa별 lambda/ridge)
        ...

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)  # (64, 2)
```

**핵심 설정:**
- action space dims: `(64, 2)` — `get_action_space_dims()` 반환값
- `dt=0.1`s, 총 예측 시간: 64 × 0.1 = 6.4초
- 정규화: `(accel - mean) / std`, `(kappa - mean) / std`
- 정규화 파라미터는 `register_buffer`로 등록 (device 이동 자동 추적)
- Tikhonov 정규화: 2차 항 사용 — jerk 자체가 아닌 jerk의 변화량 최소화

---

### is_within_bounds (L102-L123)

정규화된 action이 물리적 한계 내에 있는지 검증.

```python
def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
    # action: (..., N, 2)
    accel = action[..., 0]
    kappa = action[..., 1]
    # 역정규화 후 bounds 확인
    accel = accel * accel_std + accel_mean
    kappa = kappa * kappa_std + kappa_mean
    is_accel_within_bounds = (accel >= self.accel_bounds[0]) & (accel <= self.accel_bounds[1])
    is_kappa_within_bounds = (kappa >= self.curvature_bounds[0]) & (kappa <= self.curvature_bounds[1])
    return torch.all(is_accel_within_bounds & is_kappa_within_bounds, dim=-1)
    # 반환: (...,) — 모든 웨이포인트에서 bounds 만족 여부
```

---

### 내부 헬퍼 메서드

#### _v_to_a (L125-L160) — velocity → acceleration 변환

```python
@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
    # v: (..., N+1)
    dv = (v[..., 1:] - v[..., :-1]) / self.dt  # (..., N)
    # Tikhonov 2차 정규화: jerk smoothness 보장
    a = solve_xs_eq_y(
        s=torch.ones_like(dv),
        y=dv,
        dt=self.dt,
        lam=self.a_lambda,
        ridge=self.a_ridge,
        w_smooth1=None,
        w_smooth2=1.0,  # 2차 항 사용
        w_smooth3=None,
    )
    return a  # (..., N)
```

#### _theta_v_a_to_kappa (L162-L205) — heading+velocity → curvature 변환

```python
@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def _theta_v_a_to_kappa(self, theta, v, a) -> torch.Tensor:
    # theta: (..., N+1), v: (..., N+1), a: (..., N)
    dtheta = theta[..., 1:] - theta[..., :-1]   # (..., N)
    dt = self.dt
    s = dt * v[..., :-1] + (dt**2) / 2.0 * a    # (..., N) — 호 길이 근사
    # kappa = dtheta / s (정규화 포함)
    return solve_xs_eq_y(s=s, y=dtheta, w_smooth2=1.0, ...)  # (..., N)
```

---

### action_to_traj — 핵심 변환 (L300-L382)

Diffusion이 출력한 normalized action (accel, curvature)을 실제 trajectory (x, y, z, rotation)로 변환.

```python
def action_to_traj(
    self,
    action: torch.Tensor,          # (..., T, 2) — normalized [accel, kappa]
    traj_history_xyz: torch.Tensor, # (..., T, 3)
    traj_history_rot: torch.Tensor, # (..., T, 3, 3)
    t0_states: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    accel, kappa = action[..., 0], action[..., 1]

    # Step 1: 역정규화 (L319-326)
    accel = accel * accel_std + accel_mean
    kappa = kappa * kappa_std + kappa_mean

    # Step 2: 초기 속도 추정 (L328-331)
    if t0_states is None:
        t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
    v0 = t0_states["v"]  # (...,) — 히스토리 끝점 속도
    dt = self.dt
    dt_2_term = 0.5 * (self.dt**2)

    # Step 3: 속도 적분 (L334-341)
    velocity = torch.cat([
        v0.unsqueeze(-1),
        v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1),
    ], dim=-1)  # (..., N+1) — v[k] = v0 + Σ(a[i]*dt)

    # Step 4: 방향각(yaw) 적분 (L342-353)
    initial_yaw = torch.zeros_like(v0)
    theta = torch.cat([
        initial_yaw.unsqueeze(-1),
        initial_yaw.unsqueeze(-1)
        + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
        + torch.cumsum(kappa * accel * dt_2_term, dim=-1),
    ], dim=-1)  # (..., N+1) — θ[k] = Σ(κ*v[i]*dt) + Σ(κ*a[i]*dt²/2)

    # Step 5: 위치(x, y) 적분 — 사다리꼴 적분법 (L354-366)
    half_dt_term = 0.5 * dt
    initial_x = torch.zeros_like(v0)
    initial_y = torch.zeros_like(v0)
    x = (
        initial_x.unsqueeze(-1)
        + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term, dim=-1)
        + torch.cumsum(velocity[..., 1:]  * torch.cos(theta[..., 1:])  * half_dt_term, dim=-1)
    )  # (..., N)
    y = (
        initial_y.unsqueeze(-1)
        + torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt_term, dim=-1)
        + torch.cumsum(velocity[..., 1:]  * torch.sin(theta[..., 1:])  * half_dt_term, dim=-1)
    )  # (..., N)

    # Step 6: 출력 구성 (L367-382)
    traj_future_xyz = torch.zeros(*batch_dim, self.n_waypoints, 3, ...)
    traj_future_xyz[..., 0] = x
    traj_future_xyz[..., 1] = y
    traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]  # z는 히스토리 마지막값 유지

    traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))

    return traj_future_xyz, traj_future_rot  # (..., 64, 3), (..., 64, 3, 3)
```

**핵심 포인트:**
- Unicycle 운동학: `accel → velocity` (누적합), `velocity → position` (이중 누적합)
- yaw 적분: `κ*v*dt` (등속 항) + `κ*a*dt²/2` (가속 보정 항)
- 사다리꼴 적분: `(v[k]*cos(θ[k]) + v[k+1]*cos(θ[k+1])) * dt/2` — 수치 정확도 향상
- 출력: xyz 위치 `(..., 64, 3)` + 3D 회전행렬 `(..., 64, 3, 3)`
- z 좌표: 히스토리 마지막 z값 그대로 유지 (2D 운동학)
- `@torch.amp.autocast(enabled=False)` 미적용 — FP32 정밀도 강제 불필요 (학습/추론 공용)

---

### traj_to_action — 역변환 (L224-L298)

학습 시 GT trajectory에서 action 추출. 추론에는 사용하지 않음.

```python
@torch.no_grad()
@torch._dynamo.disable()
@torch.amp.autocast(device_type="cuda", enabled=False)
def traj_to_action(
    self,
    traj_history_xyz,  # (..., T, 3)
    traj_history_rot,  # (..., T, 3, 3)
    traj_future_xyz,   # (..., T, 3)
    traj_future_rot,   # (..., T, 3, 3)
    t0_states=None,
    output_all_states=False,
) -> torch.Tensor:

    # 히스토리 끝 + 미래 궤적 연결
    full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[..., :2]
    dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]

    # theta smoothing → v 추정 → accel → kappa 역계산
    theta = theta_smooth(traj_future_rot, ...)    # 방향각 smoothing
    v = dxy_theta_to_v(dxy, theta, v0, ...)       # (..., N+1)
    accel = self._v_to_a(v)                        # (..., N)
    kappa = self._theta_v_a_to_kappa(theta, v, accel)  # (..., N)

    # 정규화
    accel = (accel - accel_mean) / accel_std
    kappa = (kappa - kappa_mean) / kappa_std

    return torch.stack([accel, kappa], dim=-1)  # (..., N, 2)
```

**특이사항:**
- `@torch._dynamo.disable()` 적용 — torch.compile 대상에서 제외 (역변환 수치 연산의 안정성 보장)
- `@torch.amp.autocast(enabled=False)` — FP32 강제 (정밀도 손실 방지)
- `output_all_states=True` 시 `(action, [v, accel, theta])` 튜플 반환 (디버깅/분석용)

---

### estimate_t0_states (L207-L222)

히스토리 궤적에서 현재 시점(t0) 속도 추정.

```python
@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def estimate_t0_states(self, traj_history_xyz, traj_history_rot) -> dict:
    full_xy = traj_history_xyz[..., :2]           # (..., N_hist, 2)
    dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
    theta = so3_to_yaw_torch(traj_history_rot)
    theta = unwrap_angle(theta)

    v = dxy_theta_to_v_without_v0(dxy=dxy, theta=theta, ...)  # (..., N+1)
    v_t0 = v[..., -1]                            # 히스토리 마지막 속도
    return {"v": v_t0}
```
