# Alpamayo VRAM 최적화 실험

Alpamayo-R1-10B 모델의 VRAM 사용량 분석 및 최적화를 위한 실험 스크립트와 데이터를 관리하는 디렉토리입니다.

## 파일 목록

| 파일 | 설명 |
|------|------|
| `test_dummy_inference.py` | 더미 데이터를 사용한 Alpamayo-R1-10B 추론 테스트 스크립트. 모델 로드부터 추론까지 각 단계별 VRAM 사용량을 측정합니다. |
| `profile_memory.py` | GPU 메모리 프로파일링 스크립트. 0.5초 간격으로 VRAM 사용량을 모니터링하며, 5단계(데이터 생성 / 모델 로드 / GPU 이동 / 입력 준비 / 추론)별 소요 시간과 메모리를 기록합니다. |
| `memory_profile.csv` | `profile_memory.py` 실행 결과로 생성된 메모리 프로파일 데이터. 시간별 allocated/reserved VRAM과 실행 단계(phase) 정보를 포함합니다. |

## 실행 방법

alpamayo venv를 활성화한 상태에서 실행합니다:

```bash
source ../alpamayo/ar1_venv/bin/activate
# 또는
source ../alpamayo/.venv/bin/activate

python test_dummy_inference.py
python profile_memory.py
```

스크립트 내부에 `sys.path` 설정이 포함되어 있어, venv 없이도 `../alpamayo/src`의 `alpamayo_r1` 모듈을 import할 수 있습니다.
