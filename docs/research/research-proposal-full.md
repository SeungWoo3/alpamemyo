# 소비자 GPU 환경에서의 대규모 VLA 모델 On-Demand Layering 추론 최적화

## 연구 계획서

---

## 1. 연구 배경 및 동기

### 1.1 문제 정의

자율주행용 Vision-Language-Action (VLA) 모델의 규모가 급격히 증가하면서, 소비자급 GPU에서의 추론이 사실상 불가능해지는 VRAM 벽(VRAM Wall) 문제가 대두됨. 대표적으로 NVIDIA Alpamayo-R1-10B은 FP16 기준 **22.16 GB**의 파라미터 메모리를 요구하나, RTX 3080 Ti와 같은 소비자 GPU는 **12 GB VRAM**만 제공.

현재 유일한 대안인 CUDA Unified Memory의 페이지 폴트 기반 자동 스와핑은 **273.79초**의 추론 시간을 발생시키며, 이는 이론적 최적(~5.0초) 대비 **55배 느린** 수치. 추론 시간의 **98%가 순수 스와핑 오버헤드**에 해당.

### 1.2 연구 대상 모델

| 항목 | 사양 |
|------|------|
| 모델 | NVIDIA Alpamayo-R1-10B |
| 파라미터 | 11.08B (FP16 22.16 GB) |
| 아키텍처 | Vision Encoder (SigLIP, 1.15 GB) + VLM (Qwen3-VL-8B, 15.17 GB) + Expert (Flow Matching Decoder, 4.56 GB) |
| 특징 | Chain-of-Causation (CoC) 추론, 이기종 3-모듈 구조 |

### 1.3 하드웨어 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA RTX 3080 Ti (12 GB VRAM, 912 GB/s) |
| CPU | Intel i7-10700K |
| RAM | 15.55 GB |
| PCIe | Gen3 x16 (이론 15.75 GB/s, 실측 H2D 8.5 GB/s pinned) |
| OS | WSL2 (Linux 6.6.87.1) |

### 1.4 연구 제약 조건

- **모델 정확도 보존**: 양자화, 가지치기, 지식 증류 등 모델 품질 저하 기법 배제
- **시스템/메모리 레벨 최적화**에 집중: 동일 FP16 가중치로 추론 속도 극대화
- **단일 소비자 GPU** 환경 한정 (다중 GPU 기법 배제)

---

## 2. 선행 연구 분석

### 2.1 기존 오프로딩 시스템 비교

| 시스템 | H2D Prefetch | D2H | 대상 | 한계 |
|--------|:-----------:|:---:|------|------|
| DeepSpeed ZeRO-Inference | O | 불필요 | 다중 GPU | 단일 GPU 미지원 |
| FlexGen (ICML 2023) | O | O | 배치 처리 | 단일 요청 비효율 |
| PIPO (2025) | O | O | LLM 서빙 | D2H 오버헤드 존재 |
| HuggingFace Accelerate | 최근 도입 | GPU free | 범용 | 모델별 커스텀 연산 비호환 |
| vLLM | O | O | LLM 서빙 | 다중 요청 최적화 특화 |

### 2.2 기존 연구의 공백

1. **D2H 불필요성 미인식**: 대부분의 오프로딩 시스템이 파라미터 추론에서 D2H copy를 수행하나, CPU 원본이 존재하는 경우 GPU 메모리 해제만으로 충분
2. **VLA 도메인 부재**: 기존 연구가 LLM/배치 서빙에 집중, 이기종 모듈(Vision + Language + Action) 복합 구조에 대한 체계적 최적화 연구 부재
3. **소비자 GPU 실증 부족**: 대부분 A100/H100 급 환경 기준, 12 GB VRAM + PCIe Gen3 조건의 실질적 성능 분석 부족

---

## 3. 연구 목표

### 3.1 최종 목표

12 GB VRAM 소비자 GPU에서 FP16 Alpamayo-R1-10B의 추론 시간을 **CUDA Unified Memory 대비 10배 이상 개선** (273.79초 → **25초 이하**)하되, 모델 정확도 완전 보존.

### 3.2 세부 목표

| 목표 | 측정 지표 | 목표치 |
|------|----------|--------|
| G1. 추론 시간 최소화 | end-to-end 추론 시간 | ≤ 25초 |
| G2. VRAM 한도 준수 | Peak VRAM usage | ≤ 11.5 GB |
| G3. 정확도 보존 | 출력 일치도 (vs 24 GB GPU) | 100% (bitwise) |
| G4. 구조적 한계 규명 | 이론적 최적 대비 실측 비율 | 분석 및 문서화 |

---

## 4. 예비 실험 결과 (Feasibility Study)

본 연구는 8단계의 예비 실험을 통해 기술적 타당성을 확인 완료.

### 4.1 성과 요약

| 단계 | 방법 | 추론 시간 | 개선율 |
|------|------|----------|--------|
| Baseline | FP16 + CUDA Unified Memory | 273.79초 | 1.0x |
| Phase 1 | On-Demand Layering (D2H 포함) | 116.51초 | 2.35x |
| Phase 2 | On-Demand Layering (D2H 제거) | 43.38초 | 6.31x |
| **Phase 3** | **비동기 Prefetch (2-stream)** | **38.45초** | **7.12x** |

### 4.2 핵심 발견

1. **D2H 불필요성 실증**: 파라미터 추론에서 `module.to("cpu")` 대신 GPU 메모리 free만 수행 시 116.51초 → 43.38초로 **2.68배 개선**. CPU에 원본 파라미터가 존재하므로 D2H copy는 완전히 불필요.

2. **WSL2 D2H 비정상 감속**: Pageable D2H 대역폭이 2.48 GB/s로, 정상(11.8 GB/s) 대비 **4.75배 느림**. Pinned memory 사용 시 정상 복구.

3. **성능 절벽(Performance Cliff) 발견**: VRAM 10 GB → 9 GB 구간에서 추론 시간이 6.96초 → 111.36초로 **16배 급증**. 모델 크기와 VRAM 한계의 경계를 정밀하게 규명.

4. **구조적 한계 규명**: PCIe Gen3 (8.5 GB/s) vs VRAM 대역폭 (912 GB/s)의 **107배 격차**로 인해, 레이어 전송 시간(~45 ms)이 레이어 실행 시간(~0.5 ms)의 **90배**. 파이프라이닝으로 전송을 연산에 완전 은닉하는 것은 구조적으로 불가능.

### 4.3 이론적 한계 분석

```
이론적 최적 추론 시간 (스왑 없음, ≥24 GB GPU):
  VLM (256 토큰): 16.45 GB / 912 GB/s × 256 = 4.62초
  Expert (10 스텝): 4.56 GB / 912 GB/s × 10  = 0.05초
  합계: ~5.0초

12 GB VRAM + PCIe Gen3 환경 현실적 한계: ~25-40초
  (이론 최적의 5-8배 수준)
```

---

## 5. 연구 방법론

### 5.1 전체 아키텍처

```
┌──────────────────────────────────────────────────┐
│                  CPU (System RAM)                 │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ Vision  │  │  VLM Layers  │  │   Expert    │  │
│  │ Encoder │  │  (원본 보존) │  │  (원본 보존)│  │
│  │ 1.15 GB │  │  15.17 GB    │  │  4.56 GB    │  │
│  └─────────┘  └──────┬───────┘  └──────┬──────┘  │
│                      │                 │          │
│              Pinned Buffer Pool        │          │
│              (Staging Area)            │          │
└──────────────────┬─────────────────────┘──────────┘
                   │ PCIe Gen3 x16
                   │ H2D Only (D2H 제거)
                   │ Optimal Chunk: 2-8 MB
┌──────────────────▼────────────────────────────────┐
│                  GPU (12 GB VRAM)                  │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Vision  │  │ Active   │  │   Expert         │  │
│  │ Encoder │  │ VLM      │  │   (상주)          │  │
│  │ (상주)  │  │ Layers   │  │   4.56 GB         │  │
│  │ 1.15 GB │  │ (회전)   │  │                   │  │
│  └─────────┘  └──────────┘  └──────────────────┘  │
│               ┌──────────┐                         │
│               │ KV Cache │  + Activations          │
│               │ ~0.56 GB │  ~1.10 GB               │
│               └──────────┘                         │
│  Stream 0: Compute  |  Stream 1: H2D Prefetch      │
└─────────────────────────────────────────────────────┘
```

### 5.2 핵심 기법

#### 기법 1: Zero-D2H On-Demand Layering

- CPU에 원본 파라미터 보존, GPU에서는 사용 후 메모리 free만 수행
- D2H copy 완전 제거로 전송 오버헤드 **60% 감소** (실증 완료)
- Forward hook 기반 자동화로 모델 코드 최소 수정

#### 기법 2: 2-Stream Asynchronous Prefetch Pipeline

- Stream 0 (Compute): 현재 레이어 연산 실행
- Stream 1 (Transfer): 다음 레이어 H2D 프리페치
- 연산과 전송의 시간적 오버랩으로 유효 전송 시간 감소

#### 기법 3: Pinned Memory + Optimal Chunking

- Pinned memory buffer pool로 안정적 H2D 대역폭 확보 (8.5 → 12.4 GB/s)
- 2-8 MB 최적 청크 크기로 per-transfer 오버헤드 최소화
- WSL2 환경의 0.36 ms 고정 오버헤드 완화

#### 기법 4: 이기종 모듈 적응형 스케줄링

Alpamayo의 3-모듈 구조에 따른 차별화 전략:

| 모듈 | 크기 | 전략 |
|------|------|------|
| Vision Encoder | 1.15 GB | GPU 상주 (작은 크기, 1회 실행) |
| VLM (36 layers) | 15.17 GB | On-demand layering (핵심 최적화 대상) |
| Expert (36 layers) | 4.56 GB | GPU 상주 (10 스텝 반복 실행) |

- VLM 36개 레이어 중 6개는 GPU 상주, 30개는 on-demand 로딩
- VRAM 예산: Vision(1.15) + VLM 상주(2.53) + Expert(4.56) + KV/Act(1.66) = **9.90 GB** → 여유 2.10 GB를 프리페치 버퍼로 활용

---

## 6. 연구 계획

### 6.1 Phase 1: Pinned Memory 기반 최적 전송 (2주)

**목표**: Pageable → Pinned memory 전환으로 H2D 대역폭 극대화

| 항목 | 내용 |
|------|------|
| 작업 | Pinned buffer pool 구현, 최적 청크(2-8 MB) 적용 |
| 기대 효과 | H2D 대역폭 7.77 → 12.4 GB/s (1.6배) |
| 예상 추론 시간 | 43.38초 → ~30초 |
| 검증 | 추론 정확도 100% 일치 확인, VRAM ≤ 11.5 GB |

### 6.2 Phase 2: 2-Stream 비동기 파이프라인 안정화 (2주)

**목표**: 프리페치 파이프라인의 정확성 보장 및 최적화

| 항목 | 내용 |
|------|------|
| 작업 | Stream 동기화 정밀 제어, 프리페치 타이밍 최적화 |
| 기대 효과 | 연산-전송 오버랩으로 유효 전송 시간 15-20% 감소 |
| 예상 추론 시간 | ~30초 → ~25초 |
| 검증 | CUDA event 기반 프로파일링, 출력 정확도 검증 |

### 6.3 Phase 3: 디퓨전 스텝 최적화 (1주)

**목표**: Expert 모듈의 디퓨전 스텝 축소 (10 → 5)

| 항목 | 내용 |
|------|------|
| 작업 | `num_inference_steps` 조정, 궤적 품질 비교 |
| 기대 효과 | Expert 단계 50% 감소, 전체 ~1.3배 추가 개선 |
| 예상 추론 시간 | ~25초 → ~20초 |
| 검증 | 생성 궤적의 정량적 비교 (L2 distance, 궤적 형태) |

### 6.4 Phase 4: KV Cache 오프로딩 및 활성화 최적화 (2주)

**목표**: 동적 메모리 최적화로 VRAM 여유 확보 → 추가 레이어 GPU 상주

| 항목 | 내용 |
|------|------|
| 작업 | KV Cache CPU 오프로딩, Chunked Prefill |
| 기대 효과 | ~0.8 GB VRAM 절감 → 2개 추가 레이어 GPU 상주 |
| 예상 추론 시간 | 추가 ~5% 개선 |
| 검증 | 메모리 프로파일링, 추론 정확도 검증 |

### 6.5 Phase 5: 체계적 벤치마크 및 논문 작성 (3주)

**목표**: 최종 시스템의 체계적 성능 평가 및 논문 집필

| 항목 | 내용 |
|------|------|
| 작업 | Ablation study, 다양한 VRAM 조건별 성능 측정, 관련 연구 비교 |
| 산출물 | 학회 논문 (시스템 최적화 분야) |
| 비교 대상 | CUDA Unified Memory, HF Accelerate, FlexGen |

---

## 7. 기대 성과

### 7.1 정량적 목표

| 지표 | Baseline | 현재 최선 | 최종 목표 |
|------|----------|----------|----------|
| 추론 시간 | 273.79초 | 38.45초 | **≤ 25초** |
| 개선 배율 | 1.0x | 7.12x | **≥ 10x** |
| Peak VRAM | 21.52 GB | 11.03 GB | **≤ 11.5 GB** |
| 정확도 | 100% | 100% | **100%** |

### 7.2 학술적 기여

1. **D2H 불필요성 실증**: 파라미터 추론에서 D2H copy가 불필요함을 정량적으로 입증하고, 이를 활용한 Zero-D2H 오프로딩 기법 제안
2. **VLA 도메인 최적화**: 이기종 모듈(Vision + Language + Action) 구조에 대한 적응형 메모리 스케줄링 기법 최초 제시
3. **소비자 GPU 실용성**: 12 GB VRAM + PCIe Gen3 환경에서의 구조적 성능 한계 규명 및 실용적 최적화 달성
4. **재현 가능한 벤치마크**: WSL2 + RTX 3080 Ti 환경의 체계적 성능 분석 데이터셋 공개

### 7.3 실용적 가치

- 소비자 GPU에서의 대규모 VLA 모델 추론 가능성 입증
- 에지 디바이스/로보틱스 분야에서의 온디바이스 추론 활용 경로 제시
- 오픈소스 구현체 공개를 통한 연구 커뮤니티 기여

---

## 8. 리스크 및 대응 방안

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|----------|
| 25초 목표 미달성 | 중 | Phase별 중간 결과가 이미 유의미 (38.45초 → 7.12x) |
| WSL2 특이 성능 이슈 | 중 | 네이티브 Linux 환경 전환으로 ~40% 추가 개선 가능 |
| Pinned memory 시스템 RAM 부족 | 저 | 15.55 GB 중 ~4 GB를 버퍼로 할당, 충분한 여유 |
| 비동기 파이프라인 정확도 문제 | 고 | CUDA stream 동기화 포인트 엄밀 관리, bitwise 검증 |

---

## 9. 일정 요약

```
Week 1-2:  Phase 1 — Pinned Memory 최적 전송
Week 3-4:  Phase 2 — 2-Stream 비동기 파이프라인
Week 5:    Phase 3 — 디퓨전 스텝 최적화
Week 6-7:  Phase 4 — KV Cache / 활성화 최적화
Week 8-10: Phase 5 — 벤치마크 및 논문 작성
```

---

## 참고 문헌

- Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU," ICML 2023
- Aminabadi et al., "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," SC 2022
- NVIDIA, "Alpamayo: Scaling VLAs with Parameter-Efficient Expert Fusion," 2025
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," SOSP 2023
- Park et al., "PIPO: Efficient Pipelining for LLM Inference on Memory-Constrained GPUs," 2025
