# Alpamayo VRAM 최적화 실험

Alpamayo-R1-10B 모델의 VRAM 사용량 분석 및 최적화를 위한 실험 스크립트와 데이터.

---

## 디렉토리 구조

```
experiments/
├── figures/                        # 시각화 결과 이미지
│   ├── memory_timeline.png         # FP16 VRAM 시계열
│   ├── phase_breakdown.png         # FP16 Phase별 소요 시간
│   ├── theory_vs_actual.png        # 이론 스왑 vs 실제 추론 비교
│   ├── memory_timeline_4bit.png    # 4-bit VRAM 시계열
│   ├── phase_breakdown_4bit.png    # 4-bit Phase별 소요 시간
│   └── comparison_fp16_vs_4bit.png # FP16 vs 4-bit 성능 비교
├── test_dummy_inference.py         # 더미 데이터 추론 테스트
├── profile_memory.py               # FP16 메모리 프로파일링
├── profile_memory_4bit.py          # 4-bit 메모리 프로파일링
├── memory_profile.csv              # FP16 프로파일링 데이터
├── memory_profile_4bit.csv         # 4-bit 프로파일링 데이터
├── visualize_baseline.py           # FP16 시각화 스크립트
├── visualize_4bit.py               # 4-bit 시각화 + FP16 비교
└── README.md
```

## 실험 결과 요약

### Exp 1: FP16 베이스라인 (nvidia/Alpamayo-R1-10B)
- **Peak VRAM**: 21.52 GB (물리 12GB 초과 → Unified Memory 스왑)
- **VLM 추론**: 273.79초 (전체의 78.5%)
- **토큰 생성**: ~1.07초/토큰 (정상 대비 20~30배 느림)

### Exp 2: 4-bit 양자화 (dwko/Alpamayo-R1-10B-4bit, BnB fp4)
- **Peak VRAM**: 8.87 GB (12GB 이내, 스왑 없음)
- **VLM 추론**: 4.51초 (FP16 대비 **55.8배 빠름**)
- **핵심 발견**: FP16의 273초 추론은 전부 메모리 스왑 오버헤드

## 실행 방법

```bash
# alpamayo venv 활성화
source ../alpamayo/ar1_venv/bin/activate

# FP16 프로파일링
python profile_memory.py

# 4-bit 프로파일링
python profile_memory_4bit.py

# 시각화 생성
python visualize_baseline.py
python visualize_4bit.py
```

> sys.path 설정이 포함되어 있어 venv 활성화 없이도 실행 가능합니다.
