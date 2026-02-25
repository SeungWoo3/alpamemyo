# Alpamemyo — Alpamayo VRAM 최적화 연구

NVIDIA Alpamayo-R1-10B (자율주행 VLA 모델, 24GB VRAM 요구)을 12GB VRAM 환경에서 효율적으로 구동하기 위한 연구 프로젝트.

## 프로젝트 구조

```
workspace/
├── alpamayo/              # NVlabs/alpamayo 원본 소스 (클론)
│   ├── src/               # alpamayo_r1 패키지
│   ├── ar1_venv/          # Python 3.12 가상환경
│   └── ...
├── experiments/           # 실험 스크립트, 데이터, 시각화
│   ├── figures/           # 시각화 결과 이미지
│   ├── profile_memory*.py # 프로파일링 스크립트
│   ├── visualize_*.py     # 시각화 스크립트
│   └── *.csv              # 프로파일링 데이터
├── docs/                  # 프로젝트 문서
│   ├── research.md        # 연구 방향성 (살아있는 문서)
│   └── alpamayo-setup-guide.md  # 설치/실행 가이드
└── work-log/              # 일별 작업 로그
    └── 2026-02-25.md
```

## 환경

| 항목 | 스펙 |
|------|------|
| GPU | RTX 3080 Ti 12GB |
| CPU | i7-10700K |
| RAM | 16GB |
| PCIe | Gen 3 x16 (~16GB/s) |
| OS | WSL2 (Ubuntu) |
| CUDA | 12.6 |

## 핵심 실험 결과

| 모델 | Peak VRAM | 추론 시간 | 비고 |
|------|-----------|-----------|------|
| FP16 (원본) | 21.52 GB | 273.79초 | Unified Memory 스왑 발생 |
| 4-bit (BnB fp4) | 8.87 GB | 4.51초 | 12GB 이내, 스왑 없음 |

**핵심 발견**: FP16의 273초 추론 시간은 연산이 아닌 **CPU↔GPU 메모리 스왑 오버헤드** (이론 0.6초 vs 실제 273초, ~456배 차이).

## 문서

- [연구 방향성](docs/research.md) — 연구 목표, 실험 계획, 의사결정 로그
- [설치 가이드](docs/alpamayo-setup-guide.md) — WSL2 환경 설치 및 실행 방법
- [실험 README](experiments/README.md) — 실험 스크립트 및 결과 상세
