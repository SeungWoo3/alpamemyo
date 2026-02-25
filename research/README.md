# Alpamayo VRAM 최적화 연구 탐색

> 시작일: 2026-02-25

## 연구 배경
- NVIDIA Alpamayo-R1-10B: 24GB VRAM 요구 자율주행 VLA 모델
- 목표: 12GB VRAM (RTX 3080 Ti)에서 효율적 구동
- FP16 베이스라인: 추론 273.79초, Peak VRAM 21.52GB (스왑 오버헤드)
- 4-bit 양자화: 추론 4.91초, Peak VRAM 8.87GB (스왑 없음)

## 연구 방향

| # | 방향 | 디렉토리 | 상태 |
|---|------|---------|------|
| 1 | On-Demand Layering (RTSS 2022 참고) | `01-on-demand-layering/` | 탐색 중 |
| 2 | 메모리 관점 파이프라이닝 | `02-memory-pipelining/` | 탐색 중 |
| 3 | CPU-GPU 스와핑 비효율 해결 | `03-swap-optimization/` | 탐색 중 |
| 4 | 창의적 접근 (새로운 주제) | `04-creative-approaches/` | 탐색 중 |

## 선행연구
- [Alpamayo 선행연구](prior-work-alpamayo.md)
- [관련 연구 (GPU 메모리 최적화)](related-research.md)

## 디렉토리 구조
```
research/
├── README.md                    # 이 파일
├── prior-work-alpamayo.md       # 알파마요 선행연구 정리
├── related-research.md          # 관련 연구 (Demand Layering 등)
├── final-report.md              # 통합 보고서 (최종)
├── 01-on-demand-layering/       # 연구 방향 1
│   ├── analysis.md
│   └── (실험 스크립트/데이터)
├── 02-memory-pipelining/        # 연구 방향 2
│   ├── analysis.md
│   └── (실험 스크립트/데이터)
├── 03-swap-optimization/        # 연구 방향 3
│   ├── analysis.md
│   └── (실험 스크립트/데이터)
└── 04-creative-approaches/      # 연구 방향 4
    ├── analysis.md
    └── (실험 스크립트/데이터)
```
