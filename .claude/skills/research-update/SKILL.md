---
name: research-update
description: 오늘의 연구/작업 결과를 work-log에 자동 기록. 사용자가 "/research-update 실험명 결과요약" 형태로 호출.
disable-model-invocation: true
allowed-tools: Write, Read, Bash, Glob, Grep
---

# 연구 작업 로그 기록

오늘 날짜의 work-log 파일(`docs/work-log/YYYY-MM-DD.md`)에 작업 결과를 기록합니다.

## 절차

1. 현재 날짜 확인 (Bash: `date +%Y-%m-%d`)
2. `docs/work-log/` 디렉토리에 오늘 날짜 파일이 있는지 확인
3. 없으면 새로 생성, 있으면 하단에 추가
4. $ARGUMENTS에서 실험명과 결과 요약 추출
5. 관련 research 폴더의 results.json, 시각화 PNG 경로도 함께 기록

## 기록 형식

```markdown
# 작업 로그 — YYYY-MM-DD

## [HH:MM] 작업 제목
- **요청**: 사용자 요청 요약
- **상태**: 완료 / 진행중 / 실패
- **수행 내용**: 무엇을 했는지
- **결과**: 핵심 결과 (수치 포함)
- **시각화**: PNG 경로 (있는 경우)
- **커밋**: 커밋 해시 (있는 경우)
```

## 어투
- 간결한 명사형 ("실험 진행", "확인", "개선")
