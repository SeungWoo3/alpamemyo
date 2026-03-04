# Stage 4: VLM Generation

## 개요

비전 토큰과 텍스트 토큰을 결합하여 Qwen3-VL-8B 모델로 Chain-of-Thought 추론 수행 후 Expert 토큰 생성.
히스토리 trajectory를 `input_ids`에 융합한 뒤 autoregressive 방식으로 생성 진행.
생성된 KV cache가 Stage 5 Expert Decoder의 조건 정보로 전달.

**코드 위치**: `alpamayo_r1.py` L122–230

---

## 입력

### Vision Tokens
```
(N_patches / 4, 4096)
```
- Stage 3 Patch Merger 출력
- `N_patches / 4`: spatial merge ratio 4 적용 후 패치 수
- `4096`: VLM hidden dim으로 projection 완료된 상태

### Text Tokens
```
input_ids: (B, L_seq)
```
- 프롬프트 텍스트 + `<traj_hist_start>` / `<traj_hist_end>` 특수 토큰 포함
- `fuse_traj_tokens` (base_model.py L169–198): 히스토리 trajectory를 `input_ids`에 융합
  - 과거 waypoint들을 연속 토큰으로 인코딩하여 시퀀스에 삽입

---

## VLM 구조: Qwen3-VL-8B

| 항목 | 값 |
|------|-----|
| 레이어 수 | 36 Transformer layers |
| Hidden dim | 4096 |
| Attention heads | 32 (query) |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| FFN intermediate | 12288 |
| 파라미터 수 | ~8B |

**Grouped Query Attention (GQA)**: 32 query heads → 8 KV head 그룹으로 분리.
메모리 효율 향상 및 KV cache 크기 감소.

---

## 히스토리 Trajectory 융합 (fuse_traj_tokens)

**위치**: `base_model.py` L169–198

```
history trajectory (waypoints) → tokenize → input_ids에 삽입
```

- 과거 N 스텝의 (x, y, yaw) waypoint를 특수 토큰 시퀀스로 변환
- `<traj_hist_start>` ~ `<traj_hist_end>` 구간에 삽입
- VLM이 현재 위치/속도 컨텍스트를 참조 가능하도록 설계

---

## Autoregressive Generation

**위치**: `alpamayo_r1.py` L165–198

### 생성 파라미터
```python
vlm.generate(
    input_ids=input_ids,
    top_p=0.98,
    temperature=0.6,
    ...
)
```

### 특수 Logits Processors

#### ExpertLogitsProcessor
- 일반 vocabulary 토큰의 logit을 마스킹
- trajectory 관련 특수 토큰만 생성 허용
- Expert token vocabulary로 생성 공간 제한

#### StopAfterEOS
- `<traj_future_start>` 토큰 생성 시 즉시 중단
- Chain-of-Thought 추론 완료 후 Expert Decoder로 전환하는 시점

### 생성 흐름
```
[Vision tokens] + [Text + History traj tokens]
        ↓
    Qwen3-VL-8B (36 layers, CoT reasoning)
        ↓
    <traj_future_start> 토큰 생성 → 중단
        ↓
    KV cache 저장 (past_key_values)
```

---

## 출력

### KV Cache (past_key_values)
```
List[Tuple[K, V]] × 36 layers
K, V shape: (B, 8, L_seq, 128)   # 8 KV heads, head_dim=128
```

- 36개 레이어 각각의 Key, Value 행렬 저장
- Stage 5 Expert Decoder가 cross-attention 형태로 참조
- 전체 reasoning context를 Expert에게 전달하는 핵심 인터페이스

---

## 관련 파일

- [`code_references/vlm_model.md`](code_references/vlm_model.md) — VLM 추론 래퍼
- `alpamayo_r1.py` L122–230 — 전체 생성 루프
