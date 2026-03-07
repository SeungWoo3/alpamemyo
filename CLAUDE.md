# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**Carlamayo** — NVIDIA Alpamayo-R1-10B (자율주행 VLA 모델, 24GB VRAM 요구)을 12GB VRAM 환경에서 효율적으로 구동하기 위한 On-Demand Layering 최적화 연구.

- **연구 제약**: 시스템/메모리 레벨 최적화만 허용. 양자화/가지치기 등 모델 정확도를 떨어뜨리는 방식은 배제.
- **언어**: 한국어 (모든 문서, 커밋 메시지, 대화)
- **하드웨어**: RTX 3080 Ti 12GB, i7-10700K, 15.55 GB RAM, PCIe Gen3 x16, WSL2

## 실험 실행

```bash
# Alpamayo 가상환경 활성화
source alpamayo/ar1_venv/bin/activate

# 실험 스크립트 실행 (각 실험 폴더에 메인 .py 파일이 있음)
python research/{experiment_dir}/{script}.py

# 시각화 생성
python research/{experiment_dir}/create_figures.py

# MD → PDF 변환
python tools/md2pdf.py input.md
python tools/md2pdf.py --dir docs/research --recursive
```

## 핵심 규칙

### 마스터-서브 에이전트 구조
- 메인 세션: 오케스트레이터 역할만 수행
- 실제 작업: subagent(백그라운드)로 위임
- 코드/문서 생성 시 검증 에이전트 필수

### 작업 흐름
1. 요청 수신 → `docs/work-log/YYYY-MM-DD.md` 기록 시작
2. 작업 수행 (background subagent)
3. 결과 검증 (별도 검증 agent)
4. work-log 업데이트 → 유의미한 변화 시 자동 git commit

### 문서/결과물 규칙
- **문서 어투**: 간결한 명사형 ("실험 진행", "확인", "개선" — 경어체 X)
- **MD 파일 생성 시 항상 PDF도 함께 생성** (`python tools/md2pdf.py`)
- **실험 결과는 항상 시각화 자료 포함** (matplotlib 등)
- **지식/정보 질문은 `docs/info.md`에 자동 정리**

## 모델 아키텍처 (Alpamayo-R1-10B)

7-Stage 추론 파이프라인. 상세 분석은 `analysis/` 참조.

```
Stage 1-3: Vision Encoder (Qwen3-VL ViT 27L, hidden=1152 → PatchMerger → 4096)
Stage 4-5: VLM Backbone (Qwen3-VL-8B, 36L, hidden=4096, GQA 32Q/8KV)  ← 78.5% 병목
Stage 6:   Expert Decoder (hidden=2048, action projection)
Stage 7:   Diffusion (Flow Matching, 10 Euler steps → 64 waypoints)
```

- **총 VRAM 요구**: 21.52GB (FP16), 12GB 초과분은 CUDA Unified Memory가 스왑
- **핵심 병목**: VLM 36-layer autoregressive 생성 (토큰당 1.07초, 이론 대비 20-30x 느림)
- **소스 코드**: `alpamayo/src/alpamayo_r1/` (패키지), 추론 진입점 `test_inference.py`

### 주요 소스 파일
| 컴포넌트 | 경로 |
|----------|------|
| VLM + 추론 | `alpamayo/src/alpamayo_r1/models/alpamayo_r1.py` |
| Vision Encoder | `alpamayo/ar1_venv/.../transformers/models/qwen3_vl/modeling_qwen3_vl.py` |
| Diffusion | `alpamayo/src/alpamayo_r1/diffusion/flow_matching.py` |
| Action Space | `alpamayo/src/alpamayo_r1/action_space/unicycle_accel_curvature.py` |
| Config | `alpamayo/src/alpamayo_r1/config.py` |

## 현재 연구 성과

| 실험 | 추론 시간 | 대비 | 비고 |
|------|-----------|------|------|
| 베이스라인 FP16 | 273.79s | 1x | CUDA Unified Memory 스왑 |
| Demand Layering (D2H 제거) | **43.38s** | **6.31x** | 현재 최고 성과 |

**핵심 발견**: D2H 불필요 (CPU 원본 유지), WSL2 Pageable D2H 비정상, 최적 청크 2-8 MB

## 프로젝트 구조

```
workspace/
├── alpamayo/              # NVlabs/alpamayo 원본 소스 (ar1_venv 가상환경 포함)
├── analysis/              # 7-Stage 파이프라인 단계별 분석 문서
├── research/              # 실험별 폴더 (스크립트, *_results.json, 시각화 PNG)
│   ├── 01-on-demand-layering/   ~ 08-async-pipeline/
├── docs/
│   ├── work-log/          # 일별 작업 로그
│   ├── research/          # 연구 보고서, 제안서
│   ├── seminar/           # 세미나/미팅 자료
│   ├── research.md        # 연구 방향성 (살아있는 문서)
│   └── info.md            # 기술 지식 정리
└── tools/
    └── md2pdf.py          # MD→PDF 변환 (한국어 지원, WeasyPrint)
```

## 외부 연동
- **GitHub**: SeungWoo3/alpamemyo (public), 브랜치: main, gh CLI 인증 완료
- **Notion MCP**: Carlamayo 프로젝트 페이지 (`3190badf-7252-80a7-91d8-f31f9bfb80bb`) — Feasibility Test 섹션에 실험 결과 기록
- **sudo 비밀번호**: tmddn1054@ (필요 시 자동 입력)

## Custom Skills
- `/notion-sync [실험폴더]` — Notion 페이지에 실험 결과 동기화
- `/research-update 실험명 결과요약` — work-log 자동 기록
