# 추론 스크립트 코드 레퍼런스

> **원본 경로**: `alpamayo/src/alpamayo_r1/test_inference.py`

## 전체 추론 파이프라인 (L1-L78)

Alpamayo-R1-10B end-to-end 추론 스크립트. 데이터셋 로딩 → 모델 초기화 → 추론 → minADE 평가 순으로 수행.

---

### 1단계: 데이터셋 로딩 (L29-L33)

```python
# 예시 clip ID — AIAV 데이터셋 단일 시퀀스 식별자
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"

data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
# t0_us: 타임스탬프 기준점 (마이크로초 단위, 5.1초 지점)
# data 키: image_frames, ego_history_xyz, ego_history_rot, ego_future_xyz

messages = helper.create_message(data["image_frames"].flatten(0, 1))
# image_frames: 멀티카메라 프레임 시퀀스 → VLM 입력 메시지 포맷으로 변환
# flatten(0, 1): 배치+카메라 차원 병합
```

---

### 2단계: 모델 초기화 (L35-L36)

```python
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
# HuggingFace Hub에서 10B 파라미터 모델 로드
# dtype=bfloat16: VRAM 절약 (FP16 대비 수치 안정성 유리)

processor = helper.get_processor(model.tokenizer)
# 비전-언어 토크나이저+이미지 프로세서 래퍼
```

---

### 3단계: 입력 전처리 (L38-L52)

```python
# 채팅 템플릿 적용 및 토크나이징
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,   # CoC (Chain-of-Causation) 롤아웃 계속 생성
    return_dict=True,
    return_tensors="pt",
)

# 모델 입력 구성: 비전-언어 토큰 + 자아 궤적 히스토리
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],  # 과거 위치 시퀀스
    "ego_history_rot": data["ego_history_rot"],  # 과거 회전 시퀀스
}

model_inputs = helper.to_device(model_inputs, "cuda")
```

---

### 4단계: 추론 실행 (L54-L63)

```python
torch.cuda.manual_seed_all(42)  # 재현성 보장 (VLM 샘플링의 결정론적 재현)

with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,               # nucleus sampling — 확률 누적 98% 내 토큰 샘플링
        temperature=0.6,          # 낮을수록 확정적, 높을수록 다양
        num_traj_samples=1,       # 출력 trajectory 수 (VRAM 제약으로 1 사용; 늘리면 minADE 향상)
        max_generation_length=256, # CoC 텍스트 최대 토큰 수
        return_extra=True,         # CoC 텍스트 및 추가 정보 반환
    )
# pred_xyz: (batch, num_traj_sets, num_traj_samples, 64, 3)
# pred_rot: (batch, num_traj_sets, num_traj_samples, 64, 3, 3)
# extra:    {"cot": [...]} — Chain-of-Causation 텍스트
```

**핵심 포인트:**
- `sample_trajectories_from_data_with_vlm_rollout`: VLM의 CoC 텍스트 생성 + diffusion 기반 trajectory 샘플링 통합 수행
- `top_p=0.98`, `temperature=0.6`: 공식 추천값 — 다양성과 품질 균형
- `num_traj_samples=6`이 논문 평가 기준; 메모리 제약 시 1로 감소
- `torch.autocast("cuda", dtype=torch.bfloat16)`: 혼합 정밀도 추론 (내부 FP32 연산 유지)

---

### 5단계: Chain-of-Causation 출력 (L66)

```python
# extra["cot"]: VLM이 생성한 인과 추론 텍스트 (trajectory별)
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])
# 출력 형태: "The vehicle is approaching an intersection..."
# 크기: [batch_size, num_traj_sets, num_traj_samples]
```

---

### 6단계: minADE 평가 (L68-L72)

```python
# GT: (N_hist+N_future, 3) 중 미래 x,y 추출
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
# shape: (2, 64) — (xy, waypoints)

# pred: (batch, traj_sets, traj_samples, waypoints, 3) → (samples, 2, waypoints)
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
# shape: (num_traj_samples, 2, 64)

# ADE(Average Displacement Error) per sample
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
# (num_traj_samples, 2, 64) - (1, 2, 64) → norm(axis=1) → (samples, 64) → mean(-1) → (samples,)

min_ade = diff.min()  # 최소 ADE (best-of-N 평가)
print("minADE:", min_ade, "meters")
```

**minADE 계산 수식:**
```
ADE_i = mean_t( ||pred_xy_i[t] - gt_xy[t]||₂ )    for trajectory i
minADE = min_i( ADE_i )
```

---

## 전체 파이프라인 요약

```
clip_id
    │
    ▼
load_physical_aiavdataset(clip_id, t0_us)
    │  image_frames, ego_history_xyz, ego_history_rot, ego_future_xyz
    ▼
helper.create_message(image_frames)        → VLM 메시지 포맷
    │
    ▼
processor.apply_chat_template(messages)   → 토크나이징 (continue_final_message=True)
    │
    ▼
AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B")
    │  dtype=bfloat16, .to("cuda")
    ▼
model.sample_trajectories_from_data_with_vlm_rollout(
    data=model_inputs,
    top_p=0.98, temperature=0.6,
    num_traj_samples=1,
    max_generation_length=256,
)
    │  pred_xyz (B, S, K, 64, 3)
    │  pred_rot (B, S, K, 64, 3, 3)
    │  extra["cot"] — Chain-of-Causation 텍스트
    ▼
minADE 계산 → 결과 출력
```

## 주요 설정값 정리

| 파라미터 | 값 | 설명 |
|---|---|---|
| `t0_us` | `5_100_000` | 추론 시작 타임스탬프 (μs) |
| `dtype` | `torch.bfloat16` | 모델 가중치 및 연산 정밀도 |
| `top_p` | `0.98` | VLM nucleus sampling 임계값 |
| `temperature` | `0.6` | VLM 샘플링 온도 |
| `num_traj_samples` | `1` (메모리 절약) / `6` (논문 기준) | 출력 trajectory 샘플 수 |
| `max_generation_length` | `256` | CoC 텍스트 최대 토큰 수 |
| `manual_seed` | `42` | 재현성 보장 시드 |
